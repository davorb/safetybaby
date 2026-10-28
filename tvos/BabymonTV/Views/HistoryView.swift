import SwiftUI
import AVKit

struct HistoryView: View {
    @Environment(AppModel.self) private var model
    @State private var events: [ServerEvent] = []
    @State private var spans: [TimelineSpan] = []
    @State private var filter = Filter.all
    @State private var selected: ServerEvent?
    @State private var playing: URL?

    enum Filter: String, CaseIterable {
        case all = "All", alerts = "Alerts", position = "Position", sleep = "Sleep", camera = "Camera"
        func matches(_ e: ServerEvent) -> Bool {
            switch self {
            case .all: true
            case .alerts: e.event.type == "alert"
            case .position: e.event.type == "vision"
            case .sleep: e.event.type == "sleep_changed"
            case .camera: e.event.type == "camera"
            }
        }
    }

    private var shown: [ServerEvent] { events.filter(filter.matches) }

    var body: some View {
        ZStack {
            Theme.ground.ignoresSafeArea()
            VStack(alignment: .leading, spacing: 24) {
                HStack(spacing: 28) {
                    Text("History").font(.ui(72, bold: true))
                    Spacer()
                    ForEach(Filter.allCases, id: \.self) { f in
                        Button { filter = f } label: {
                            Text(f.rawValue).font(.ui(32, bold: true)).padding(.horizontal, 28).frame(height: 64)
                                .foregroundStyle(filter == f ? Color(hex: 0x17140F) : Theme.text)
                                .background(Capsule().fill(filter == f ? Theme.text : Theme.raised))
                        }.buttonStyle(FocusStyle(radius: 32))
                    }
                }
                TimelineBar(spans: spans)
                HStack(alignment: .top, spacing: 40) {
                    ScrollView {
                        LazyVStack(spacing: 10) {
                            if !model.missed.isEmpty && filter == .all || filter == .alerts {
                                ForEach(model.missed) { row($0, missed: true) }
                            }
                            ForEach(shown) { row($0) }
                            if shown.isEmpty { Text("Nothing in the last 24 hours.").font(.ui(32)).foregroundStyle(Theme.muted).padding(40) }
                        }.padding(.vertical, 20).padding(.horizontal, 12)
                    }
                    .frame(maxWidth: .infinity)
                    detail.frame(width: 720)
                }
            }
            .padding(.horizontal, 20)
        }
        .task { await load() }
        .onExitCommand { model.screen = .monitor; model.dockOpen = true }
        .fullScreenCover(item: $playing) { url in ClipPlayer(url: url) }
    }

    private func row(_ e: ServerEvent, missed: Bool = false) -> some View {
        let (title, sub, tag) = Self.describe(e)
        return Button { selected = e } label: {
            HStack(spacing: 24) {
                Text(Format.clock(e.event.at)).font(.ui(30)).foregroundStyle(Theme.muted).frame(width: 100, alignment: .leading)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.ui(34, bold: true)).lineLimit(1)
                    Text(sub).font(.ui(26)).foregroundStyle(Theme.muted).lineLimit(1)
                }
                Spacer()
                Text(missed ? "MISSED" : tag).font(.ui(24, bold: true)).foregroundStyle(Theme.muted)
            }
            .foregroundStyle(Theme.text)
            .padding(.horizontal, 28).frame(height: 96)
            .background(RoundedRectangle(cornerRadius: 22).fill(selected?.id == e.id ? Theme.raised : Theme.panel))
        }
        .buttonStyle(FocusStyle(radius: 22))
    }

    @ViewBuilder private var detail: some View {
        if let e = selected ?? shown.first {
            VStack(alignment: .leading, spacing: 20) {
                if let snap = e.snapshotURL.flatMap({ model.api?.absolute($0) }) {
                    AsyncImage(url: snap) { $0.resizable().aspectRatio(contentMode: .fit) } placeholder: { Theme.panel }
                        .frame(maxWidth: .infinity).clipShape(RoundedRectangle(cornerRadius: 20))
                    Text("Snapshot \(e.event.at.formatted(.dateTime.hour().minute().second()))").font(.ui(28)).foregroundStyle(Theme.muted)
                }
                let (title, sub, _) = Self.describe(e)
                Text(title).font(.ui(40, bold: true))
                Text(sub).font(.ui(30)).foregroundStyle(Theme.muted)
                if let clip = e.clipURL.flatMap({ model.api?.absolute($0) }) {
                    Button { playing = clip } label: {
                        Label("Play clip from \(e.event.at.formatted(.dateTime.hour().minute().second()))", systemImage: "play.fill")
                            .font(.ui(36, bold: true)).foregroundStyle(Color(hex: 0x17140F))
                            .padding(.horizontal, 40).frame(height: 96).background(Capsule().fill(Theme.text))
                    }.buttonStyle(FocusStyle(radius: 48))
                } else {
                    Text("No recording covers this moment.").font(.ui(28)).foregroundStyle(Theme.muted)
                }
            }
            .padding(32).frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 28).fill(Theme.panel))
        }
    }

    private func load() async {
        guard let api = model.api else { return }
        async let t: Void = {
            while !Task.isCancelled {
                if let s = try? await api.timeline() { spans = s }
                try? await Task.sleep(for: .seconds(60))
            }
        }()
        while !Task.isCancelled {
            if let e = try? await api.events(hours: 24) { events = e }
            try? await Task.sleep(for: .seconds(15))
        }
        await t
    }

    static func describe(_ e: ServerEvent) -> (String, String, String) {
        let x = e.event
        switch x.type {
        case "sleep_changed":
            return ("\((x.previous ?? "").capitalized) to \(x.current ?? "")", "Sleep state changed", "SLEEP")
        case "vision":
            let o = x.observation
            let parts = [o.map { PositionCard.body($0.body) }, o.map { "eyes \($0.eyes)" }, o.map { "\(Int($0.confidence * 100)) %" }]
            let covered = o?.faceCovered == true
            return (covered ? "Face covered" : "Position check", parts.compactMap { $0 }.joined(separator: " · "), "POSITION")
        case "alert":
            let a = x.alert
            return (a?.title ?? "Alert", "\((a?.priority ?? "").capitalized) · " + (x.delivered == false ? "push failed" : "push sent"), "ALERT")
        case "controls":
            return ("Controls changed", "Armed \(x.controls?.armed == true ? "on" : "off") · nap mode \(x.controls?.napMode == true ? "on" : "off")", "CONTROLS")
        case "camera":
            return (x.up == true ? "Camera back online" : "Camera offline", x.detail ?? "", "CAMERA")
        default:
            return (x.type, "", "")
        }
    }
}

extension URL: @retroactive Identifiable { public var id: String { absoluteString } }

struct TimelineBar: View {
    let spans: [TimelineSpan]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Last 24 hours").font(.ui(26)).foregroundStyle(Theme.muted)
                Spacer()
                ForEach(["asleep", "stirring", "awake", "crying"], id: \.self) { s in
                    HStack(spacing: 8) {
                        Circle().fill(Theme.sleep(s)).frame(width: 16, height: 16)
                        Text(s.capitalized).font(.ui(26)).foregroundStyle(Theme.muted)
                    }
                }
            }
            GeometryReader { geo in
                let end = Date(), start = end.addingTimeInterval(-86400)
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 8).fill(Theme.raised)
                    ForEach(Array(spans.enumerated()), id: \.offset) { _, sp in
                        let a = max(sp.start.timeIntervalSince(start), 0) / 86400
                        let b = min(sp.end.timeIntervalSince(start), 86400) / 86400
                        Rectangle().fill(Theme.sleep(sp.state ?? ""))
                            .overlay { if sp.state == "crying" { Hatch(color: .black.opacity(0.35)) } }
                            .frame(width: max(2, geo.size.width * (b - a)))
                            .offset(x: geo.size.width * a)
                    }
                }.clipShape(RoundedRectangle(cornerRadius: 8))
            }.frame(height: 44)
        }
    }
}

struct ClipPlayer: View {
    let url: URL
    @State private var player = AVPlayer()

    var body: some View {
        VideoPlayer(player: player)
            .ignoresSafeArea()
            .onAppear {
                player.replaceCurrentItem(with: AVPlayerItem(url: url.removingFragment))
                if let t = url.fragment?.split(separator: "=").last.flatMap({ Double($0) }) {
                    player.seek(to: CMTime(seconds: t, preferredTimescale: 600))
                }
                player.play()
            }
            .onDisappear { player.pause() }
    }
}

extension URL {
    var removingFragment: URL {
        var c = URLComponents(url: self, resolvingAgainstBaseURL: true)
        c?.fragment = nil
        return c?.url ?? self
    }
}
