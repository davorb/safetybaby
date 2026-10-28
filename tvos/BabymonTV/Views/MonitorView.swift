import SwiftUI

struct MonitorView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ZStack {
            CameraView().ignoresSafeArea()
            LinearGradient(colors: [.black.opacity(0.85), .clear], startPoint: .top, endPoint: .bottom)
                .frame(height: 260).frame(maxHeight: .infinity, alignment: .top).ignoresSafeArea()
            LinearGradient(colors: [.black.opacity(0.85), .clear], startPoint: .bottom, endPoint: .top)
                .frame(height: 480).frame(maxHeight: .infinity, alignment: .bottom).ignoresSafeArea()

            if !model.dockOpen && model.dialog == nil && !(model.emergency != nil && model.emergencyFullScreen) {
                Button { model.dockOpen = true; model.touch() } label: { Color.clear }
                    .buttonStyle(NoFocusStyle())
                    .ignoresSafeArea()
                    .onMoveCommand { _ in model.dockOpen = true; model.touch() }
            }

            overlay
                .opacity(model.dimmed ? 0.6 : 1)
                .offset(x: model.drift, y: model.drift)
                .animation(.easeInOut(duration: 2), value: model.dimmed)

            if model.dockOpen {
                Color.black.opacity(0.62).ignoresSafeArea().transition(.opacity)
                DockView().transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.3), value: model.dockOpen)
        .animation(.easeOut(duration: 0.3), value: model.highAlerts)
        .onExitCommand { if model.dockOpen { model.dockOpen = false } }
    }

    private var overlay: some View {
        VStack(alignment: .leading, spacing: 20) {
            topRow
            if let n = model.notice { NoticeStrip(text: n.text) }
            if let t = model.toast { NoticeStrip(text: t.text) }
            if let away = model.awayFor { awayLine(away) }
            if model.connection == .live { PendingCards() }
            Spacer(minLength: 0)
            if !model.dockOpen { bottomRow }
        }
        .overlay(alignment: .topLeading) {
            if model.dockOpen, let s = model.state {
                SleepBlock(state: s.sleep, seconds: s.sleepForS, compact: true).padding(.top, 120)
            }
        }
    }

    private var topRow: some View {
        VStack(alignment: .leading, spacing: 26) {
            ForEach(model.highAlerts) { HighBanner(alert: $0) }
            HStack(alignment: .top) {
                ChipRow()
                Spacer()
                VStack(alignment: .trailing, spacing: 6) {
                    Text(model.now.formatted(.dateTime.hour(.twoDigits(amPM: .omitted)).minute(.twoDigits)))
                        .font(.ui(60, bold: true))
                    Text(frameLabel).font(.ui(26))
                        .foregroundStyle((model.frameAge ?? 0) > 10 ? Theme.high : Theme.muted)
                }.foregroundStyle(Theme.text)
            }
            if let banner = DegradedBanner.pick(model) { banner }
        }
    }

    private var frameLabel: String {
        guard let age = model.frameAge else { return "No frame yet" }
        return "Frame \(Format.duration(age)) ago"
    }

    private func awayLine(_ away: Double) -> some View {
        HStack {
            Text("Away for \(Format.duration(away)). \(model.missed.count) alert\(model.missed.count == 1 ? "" : "s") missed. See History.")
                .font(.ui(30)).foregroundStyle(Theme.text)
            Button("OK") { model.awayFor = nil }.buttonStyle(FocusStyle(radius: 20))
        }
        .padding(.horizontal, 28).padding(.vertical, 14)
        .background(Capsule().fill(Theme.normal))
    }

    private var bottomRow: some View {
        HStack(alignment: .bottom) {
            if let s = model.state, model.connection != .starting {
                SleepBlock(state: s.sleep, seconds: s.sleepForS)
                    .opacity(model.connection == .unreachable ? 0.5 : 1)
            } else if model.connection == .starting {
                StartingBlock()
            }
            Spacer()
            if model.state != nil && model.connection != .starting { PositionCard() }
        }
    }
}

struct NoFocusStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View { configuration.label }
}

struct StartingBlock: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("babymon is starting").font(.ui(84, bold: true)).foregroundStyle(Theme.text)
            Text("The service is warming up. Checking every 2 s.").font(.ui(40)).foregroundStyle(Theme.muted)
        }
    }
}

struct HighBanner: View {
    let alert: ActiveAlert
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 36) {
            Group {
                if let url = alert.snapshotURL {
                    AsyncImage(url: url) { $0.resizable().aspectRatio(contentMode: .fill) } placeholder: { Theme.panel }
                } else if let frame = model.frame {
                    Image(uiImage: frame).resizable().aspectRatio(contentMode: .fill)
                } else { Theme.panel }
            }
            .frame(width: 256, height: 144).clipShape(RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(Theme.highText, lineWidth: 4))

            VStack(alignment: .leading, spacing: 4) {
                Text("HIGH ALERT · \(Format.clock(alert.at))").font(.ui(28, bold: true)).tracking(2)
                Text(alert.title).font(.ui(62, bold: true)).lineLimit(1).minimumScaleFactor(0.6)
                Text(alert.delivered ? "Push sent to both phones" : "Push failed. The phones may not know.")
                    .font(.ui(32))
            }
            Spacer(minLength: 0)
            VStack(alignment: .trailing, spacing: 12) {
                Button { model.dismiss(alert) } label: {
                    Text("Dismiss").font(.ui(36, bold: true)).foregroundStyle(Color(hex: 0xFFD58A))
                        .padding(.horizontal, 36).frame(height: 76)
                        .background(Capsule().fill(Theme.highText))
                }.buttonStyle(FocusStyle(radius: 38))
                Text("Clears on its own when calm").font(.ui(28, bold: true))
            }
        }
        .foregroundStyle(Theme.highText)
        .padding(.leading, 24).padding(.trailing, 44).frame(height: 208)
        .background(RoundedRectangle(cornerRadius: 36).fill(Theme.high))
        .shadow(color: .black.opacity(0.55), radius: 35, y: 24)
        .transition(.move(edge: .top).combined(with: .opacity))
    }
}

/// One banner at a time, in the design's order: unreachable, camera down, vision off, push off.
enum DegradedBanner {
    @MainActor
    static func pick(_ model: AppModel) -> AnyView? {
        let list = model.degradations
        guard let first = list.first else { return nil }
        let more = list.count - 1
        switch first {
        case .unreachable:
            let last = model.lastContact.map { Format.ago(model.now.timeIntervalSince($0)) } ?? "never"
            return AnyView(card(red: true, kicker: "NOT MONITORING", title: "Can't reach babymon",
                                body: "Last contact \(last) · trying again every 10 s"
                                + (model.now.timeIntervalSince(model.lastContact ?? model.now) > 60 ? ". Phone push may still work." : ""), more: more))
        case .cameraDown:
            let down = model.state.map { Format.duration($0.cameraDownForS) } ?? ""
            return AnyView(card(red: false, kicker: "HIGH ALERT · CAMERA", title: "Camera offline for \(down)",
                                body: "No new pictures. Position checks are paused.", more: more))
        case .visionOff:
            return AnyView(card(red: false, kicker: "VISION DISABLED", title: "Position checks are off",
                                body: "Face covered, stomach and head-left alerts cannot fire.", more: more))
        case .pushOff:
            return AnyView(card(red: false, kicker: "PUSH DISABLED", title: "Phone alerts are off",
                                body: "This TV is the only place alerts will show.", more: more))
        }
    }

    private static func card(red: Bool, kicker: String, title: String, body: String, more: Int) -> some View {
        HStack(spacing: 24) {
            Image(systemName: red ? "exclamationmark.octagon.fill" : "exclamationmark.triangle.fill").font(.system(size: 44))
            VStack(alignment: .leading, spacing: 2) {
                Text(kicker).font(.ui(24, bold: true)).tracking(2)
                Text(title).font(.ui(42, bold: true))
                Text(body).font(.ui(28))
            }
            Spacer()
            if more > 0 { Text("+ \(more) more").font(.ui(28, bold: true)).padding(.horizontal, 20).padding(.vertical, 8).background(Capsule().strokeBorder(lineWidth: 3)) }
        }
        .foregroundStyle(red ? Theme.emergencyText : Theme.highText)
        .padding(.horizontal, 32).padding(.vertical, 20)
        .background(RoundedRectangle(cornerRadius: 28).fill(red ? Theme.emergencyGround : Theme.high))
        .overlay(RoundedRectangle(cornerRadius: 28).strokeBorder(red ? Theme.emergencyEdge : .clear, lineWidth: 4))
    }
}
