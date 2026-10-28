import SwiftUI
import Observation

struct ActiveAlert: Identifiable, Equatable {
    let id: Int
    let kind: String
    let tier: Tier
    let title: String
    let message: String
    let at: Date
    let snapshotURL: URL?
    let delivered: Bool
}

struct Notice: Identifiable, Equatable {
    let id = UUID()
    let text: String
}

enum Dialog: Equatable { case disarm, reference }
enum Screen { case monitor, history, settings }
enum Connection { case live, starting, unreachable }
enum ControlKind { case armed, napMode }

enum Degradation: Equatable {
    case unreachable, cameraDown, visionOff, pushOff
}

/// Single owner of everything the screens show. Polling rules follow the design's "Polling and data" board.
@MainActor
@Observable
final class AppModel {
    static let defaultsKey = "serverURL"
    static let missedWindow: TimeInterval = 120

    var serverURL: URL? = UserDefaults.standard.string(forKey: AppModel.defaultsKey).flatMap(URL.init(string:))

    var state: Situation?
    var connection: Connection = .live
    var lastContact: Date?
    var frame: UIImage?
    var frameFailed = false
    var frameAt: Date?

    var screen: Screen = .monitor
    var dockOpen = false
    var dialog: Dialog?
    var now = Date()
    var lastInput = Date()

    var emergency: ActiveAlert?
    var emergencyFullScreen = true
    var emergencyMutedUntil: Date?
    var highAlerts: [ActiveAlert] = []
    var notice: Notice?
    var toast: Notice?
    var missed: [ServerEvent] = []
    var awayFor: TimeInterval?
    var pendingToggles: [ControlKind: (target: Bool, deadline: Date)] = [:]

    @ObservationIgnored private let sound = AlertSound()
    @ObservationIgnored private var tasks: [Task<Void, Never>] = []
    @ObservationIgnored private var misses = 0
    @ObservationIgnored private var highestEventID = 0
    @ObservationIgnored private var eventsKick = false
    @ObservationIgnored private var nextEmergencyTone = Date.distantPast
    @ObservationIgnored private var highChimes: [Int: (first: Date, repeated: Bool)] = [:]
    @ObservationIgnored private var unreachableChimed = false
    @ObservationIgnored private var backgroundedAt: Date?
    @ObservationIgnored private var launched = false

    var api: APIClient? { serverURL.map(APIClient.init) }

    // MARK: lifecycle

    func connect(to url: URL) {
        serverURL = url
        UserDefaults.standard.set(url.absoluteString, forKey: Self.defaultsKey)
        restart()
    }

    func forgetServer() {
        stop()
        UserDefaults.standard.removeObject(forKey: Self.defaultsKey)
        serverURL = nil
        state = nil
        frame = nil
    }

    func restart() {
        stop()
        guard let api else { return }
        launched = false
        highestEventID = 0
        tasks = [
            Task { await stateLoop(api) },
            Task { await frameLoop(api) },
            Task { await eventsLoop(api) },
            Task { await ticker() },
        ]
    }

    func stop() {
        tasks.forEach { $0.cancel() }
        tasks = []
        sound.stop()
    }

    func scenePhaseChanged(active: Bool) {
        if active {
            if let backgroundedAt, Date().timeIntervalSince(backgroundedAt) > 10 {
                awayFor = Date().timeIntervalSince(backgroundedAt)
                eventsKick = true
            }
            backgroundedAt = nil
            touch()
        } else if backgroundedAt == nil {
            backgroundedAt = Date()
        }
    }

    func touch() { lastInput = Date() }

    // MARK: polling

    private func stateLoop(_ api: APIClient) async {
        while !Task.isCancelled {
            do {
                apply(try await api.state())
            } catch APIError.unavailable {
                misses = 0
                connection = .starting
                lastContact = Date()
            } catch {
                misses += 1
                if misses >= 3 { connection = .unreachable }
            }
            let wait: Double = misses < 4 ? 2 : misses == 4 ? 5 : 10
            try? await Task.sleep(for: .seconds(wait))
        }
    }

    private func frameLoop(_ api: APIClient) async {
        try? await Task.sleep(for: .seconds(1))
        while !Task.isCancelled {
            var wait = 2.0
            do {
                let data = try await api.frame()
                if let image = await Task.detached(priority: .utility, operation: { UIImage(data: data)?.preparingForDisplay() }).value {
                    withAnimation(.easeInOut(duration: 0.4)) { frame = image }
                    frameAt = Date()
                    frameFailed = false
                }
            } catch APIError.unavailable {
                frameFailed = true
                wait = 10
            } catch {
                if connection != .live { frameFailed = true }
            }
            try? await Task.sleep(for: .seconds(wait))
        }
    }

    private func eventsLoop(_ api: APIClient) async {
        var lastFetch = Date.distantPast
        while !Task.isCancelled {
            if eventsKick || Date().timeIntervalSince(lastFetch) >= 4 {
                eventsKick = false
                lastFetch = Date()
                await fetchEvents(api)
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
    }

    private func fetchEvents(_ api: APIClient) async {
        do {
            if !launched {
                let events = try await api.events(hours: 1)
                highestEventID = events.map(\.id).max() ?? 0
                launched = true
                ingest(events)
            } else {
                ingest(try await api.events(hours: 1, sinceID: highestEventID))
            }
        } catch {}
    }

    private func ticker() async {
        while !Task.isCancelled {
            now = Date()
            tick()
            try? await Task.sleep(for: .seconds(1))
        }
    }

    // MARK: state

    private func apply(_ s: Situation) {
        let old = state
        let recovered = connection == .unreachable
        state = s
        connection = .live
        misses = 0
        unreachableChimed = false
        lastContact = Date()
        if recovered || old.map({ changed($0, s) }) == true { eventsKick = true }

        for (toggle, p) in pendingToggles where value(of: toggle, in: s) == p.target {
            pendingToggles[toggle] = nil
        }
        clearResolvedAlerts(s)
    }

    private func changed(_ a: Situation, _ b: Situation) -> Bool {
        a.sleep != b.sleep || a.faceCoveredStreak != b.faceCoveredStreak || a.stomachStreak != b.stomachStreak
            || a.cameraUp != b.cameraUp || a.controls != b.controls
    }

    private func value(of toggle: ControlKind, in s: Situation) -> Bool {
        toggle == .armed ? s.controls.armed : s.controls.napMode
    }

    private func holds(_ kind: String, _ s: Situation) -> Bool {
        switch kind {
        case "face_covered": s.faceCoveredStreak > 0
        case "on_stomach": s.stomachStreak > 0
        case "head_left": s.headLeftForS > 0
        case "crying": s.sleep == "crying"
        case "camera_offline": !s.cameraUp
        default: false
        }
    }

    private func clearResolvedAlerts(_ s: Situation) {
        if let e = emergency, !holds(e.kind, s) { clearEmergency() }
        highAlerts.removeAll { !holds($0.kind, s) }
    }

    // MARK: alerts

    private func ingest(_ events: [ServerEvent]) {
        for e in events.sorted(by: { $0.id < $1.id }) {
            highestEventID = max(highestEventID, e.id)
            guard e.event.type == "alert", let info = e.event.alert else { continue }
            if Date().timeIntervalSince(info.at) > Self.missedWindow {
                missed.insert(e, at: 0)
                continue
            }
            let alert = ActiveAlert(
                id: e.id, kind: info.kind, tier: Tier(info.priority), title: info.title,
                message: info.message, at: info.at,
                snapshotURL: e.snapshotURL.flatMap { api?.absolute($0) },
                delivered: e.event.delivered ?? true
            )
            show(alert)
        }
    }

    func show(_ alert: ActiveAlert) {
        switch alert.tier {
        case .emergency:
            emergency = alert
            emergencyFullScreen = true
            emergencyMutedUntil = nil
            nextEmergencyTone = Date()
            dockOpen = false
            dialog = nil
        case .high:
            highAlerts.removeAll { $0.kind == alert.kind }
            highAlerts.append(alert)
            highChimes[alert.id] = (Date(), false)
            sound.play(.high)
        case .normal:
            post(notice: alert.title)
        }
    }

    func post(notice text: String) {
        let n = Notice(text: text)
        notice = n
        Task {
            try? await Task.sleep(for: .seconds(12))
            if notice == n { withAnimation(.easeOut(duration: 0.2)) { notice = nil } }
        }
    }

    func showToast(_ text: String) {
        let n = Notice(text: text)
        toast = n
        Task {
            try? await Task.sleep(for: .seconds(4))
            if toast == n { toast = nil }
        }
    }

    func clearEmergency() {
        emergency = nil
        emergencyFullScreen = true
        emergencyMutedUntil = nil
    }

    func backToCamera() { emergencyFullScreen = false }

    func muteEmergency() { emergencyMutedUntil = Date().addingTimeInterval(60) }

    func dismiss(_ alert: ActiveAlert) {
        highAlerts.removeAll { $0.id == alert.id }
        highChimes[alert.id] = nil
    }

    private func tick() {
        if let e = emergency, emergencyMutedUntil.map({ now >= $0 }) ?? true, now >= nextEmergencyTone {
            _ = e
            sound.play(.emergency)
            nextEmergencyTone = now.addingTimeInterval(8)
        }
        for alert in highAlerts {
            if let c = highChimes[alert.id], !c.repeated, now.timeIntervalSince(c.first) >= 60 {
                highChimes[alert.id] = (c.first, true)
                sound.play(.high)
            }
        }
        if connection == .unreachable, !unreachableChimed, let last = lastContact, now.timeIntervalSince(last) > 60 {
            unreachableChimed = true
            sound.play(.high)
        }
        for (toggle, p) in pendingToggles where now >= p.deadline {
            pendingToggles[toggle] = nil
            post(notice: "Couldn't change \(toggle == .armed ? "Armed" : "Nap mode"). The server did not confirm.")
        }
        if dockOpen, dialog == nil, now.timeIntervalSince(lastInput) > 12 { dockOpen = false }
    }

    func playTest(_ tone: AlertSound.Tone) { sound.play(tone) }

    // MARK: controls

    func set(_ toggle: ControlKind, to value: Bool) {
        guard let api else { return }
        pendingToggles[toggle] = (value, Date().addingTimeInterval(6))
        Task {
            do {
                try await toggle == .armed ? api.setControls(armed: value) : api.setControls(napMode: value)
            } catch {
                pendingToggles[toggle] = nil
                post(notice: "Couldn't reach babymon to change that.")
            }
        }
    }

    /// Turning Armed off asks first. Everything else is immediate.
    func requestToggle(_ toggle: ControlKind) {
        guard let s = state else { return }
        let current = value(of: toggle, in: s)
        if toggle == .armed, current { dialog = .disarm } else { set(toggle, to: !current) }
    }

    func confirmReference() async {
        guard let api else { return }
        do {
            try await api.setReference()
            showToast("Reference saved")
        } catch {
            post(notice: "No frame to save. The camera may be down.")
        }
    }

    // MARK: derived

    var dimmed: Bool { now.timeIntervalSince(lastInput) > 600 }

    /// Overlay drift against burn-in: 6 px every 5 minutes, cycling.
    var drift: CGFloat {
        [0, 6, 0, -6][Int(now.timeIntervalSince1970 / 300) % 4]
    }

    var degradations: [Degradation] {
        var out: [Degradation] = []
        if connection == .unreachable { out.append(.unreachable) }
        if let s = state {
            if !s.cameraUp { out.append(.cameraDown) }
            if !s.capabilities.vision { out.append(.visionOff) }
            if !s.capabilities.push { out.append(.pushOff) }
        }
        return out
    }

    var frameAge: Double? { frameAt.map { now.timeIntervalSince($0) } }
}
