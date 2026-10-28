import SwiftUI

enum DockItem: Hashable { case armed, nap, history, reference, settings }

struct DockView: View {
    @Environment(AppModel.self) private var model
    @FocusState private var focus: DockItem?

    var body: some View {
        VStack {
            if model.state?.controls.armed == false { DisarmedRibbon() }
            Spacer()
            HStack(spacing: 28) {
                toggleCard(.armed)
                toggleCard(.nap)
                simple(.history, "clock", "History", "Last 24 h, events and clips") {
                    model.screen = .history; model.dockOpen = false
                }
                simple(.reference, "viewfinder", "Set reference", "Save this frame as the good position") {
                    model.dialog = .reference
                }
                simple(.settings, "slider.horizontal.3", "Settings", "Server, sound, screen") {
                    model.screen = .settings; model.dockOpen = false
                }
            }
            .focusSection()
            if let hint = focus.flatMap(Self.hint) {
                Text(hint).font(.ui(30)).foregroundStyle(Theme.text)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 8)
                    .transition(.opacity)
            }
            Spacer().frame(height: 10)
        }
        .onAppear { focus = .history }
        .simultaneousGesture(TapGesture().onEnded { model.touch() })
        .onMoveCommand { _ in model.touch() }
    }

    static func hint(_ item: DockItem) -> String? {
        switch item {
        case .armed:
            "Armed: alerts are pushed to both phones. Off keeps watching and logging, but no phone is alerted."
        case .nap:
            "Nap mode adds one alert: a quiet push when the baby wakes up and stays awake, so you know the nap is over. Crying and safety alerts work with it on or off."
        default: nil
        }
    }

    private func toggleCard(_ item: DockItem) -> some View {
        let isArmed = item == .armed
        let s = model.state
        let on = isArmed ? (s?.controls.armed ?? false) : (s?.controls.napMode ?? false)
        let kind: ControlKind = isArmed ? .armed : .napMode
        let pending = model.pendingToggles[kind] != nil
        let tint = isArmed ? (on ? Theme.ok : Theme.high) : Theme.nap
        return Button { model.requestToggle(kind) } label: {
            VStack(alignment: .leading) {
                Text(isArmed ? "Armed" : "Nap mode").font(.ui(38, bold: true))
                Text(on ? "ON" : "OFF").font(.ui(78, bold: true))
                    .foregroundStyle(on || isArmed ? tint : Theme.muted).padding(.top, 2)
                Spacer(minLength: 0)
                HStack {
                    Text(isArmed ? (on ? "Alerts on" : "Alerts off") : (on ? "Alerts when awake" : "No wake-up alert"))
                        .font(.ui(26)).foregroundStyle(Theme.muted).lineLimit(2).minimumScaleFactor(0.7)
                    Spacer()
                    ZStack(alignment: on ? .trailing : .leading) {
                        Capsule().fill(on ? tint : Theme.raised).frame(width: 120, height: 64)
                        Circle().fill(on ? Theme.ground : tint).frame(width: 50, height: 50).padding(7)
                    }
                    .overlay(Capsule().strokeBorder(Theme.focus, lineWidth: pending ? 4 : 0))
                }
            }
            .foregroundStyle(Theme.text)
            .padding(28).frame(maxWidth: .infinity).frame(height: 300, alignment: .topLeading)
            .background(RoundedRectangle(cornerRadius: 32).fill(Theme.panel))
            .overlay { if isArmed && !on { Hatch(color: Theme.high.opacity(0.18)).clipShape(RoundedRectangle(cornerRadius: 32)) } }
        }
        .buttonStyle(FocusStyle())
        .focused($focus, equals: item)
    }

    private func simple(_ item: DockItem, _ symbol: String, _ title: String, _ sub: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(alignment: .leading) {
                Image(systemName: symbol).font(.system(size: 56))
                Spacer(minLength: 0)
                Text(title).font(.ui(38, bold: true))
                Text(sub).font(.ui(28)).foregroundStyle(Theme.muted).padding(.top, 6)
            }
            .foregroundStyle(Theme.text)
            .padding(28).frame(maxWidth: .infinity, alignment: .leading).frame(height: 300)
            .background(RoundedRectangle(cornerRadius: 32).fill(Theme.panel))
        }
        .buttonStyle(FocusStyle())
        .focused($focus, equals: item)
    }
}

struct DisarmedRibbon: View {
    var body: some View {
        HStack {
            Label("DISARMED · no alerts will be sent", systemImage: "shield.slash").font(.ui(48, bold: true))
            Spacer()
            Text("Camera and history still run").font(.ui(32, bold: true))
        }
        .foregroundStyle(Theme.highText)
        .padding(.horizontal, 40).frame(height: 104)
        .background(Theme.high)
        .overlay(Hatch(color: Color.black.opacity(0.08)))
        .clipShape(RoundedRectangle(cornerRadius: 24))
    }
}

/// Confirmation for disarming and for replacing the reference. Cancel is focused first.
struct DialogView: View {
    let dialog: Dialog
    @Environment(AppModel.self) private var model
    @FocusState private var focus: Bool

    var body: some View {
        ZStack {
            Color.black.opacity(0.78).ignoresSafeArea()
            VStack(alignment: .leading, spacing: 36) {
                Text(dialog == .disarm ? "Turn alerts off?" : "Replace the reference position?")
                    .font(.ui(68, bold: true))
                HStack(alignment: .top, spacing: 56) {
                    if dialog == .reference, let frame = model.frame {
                        VStack(alignment: .leading, spacing: 14) {
                            Image(uiImage: frame).resizable().aspectRatio(16 / 9, contentMode: .fill)
                                .frame(width: 560, height: 315).clipShape(RoundedRectangle(cornerRadius: 20))
                                .overlay(RoundedRectangle(cornerRadius: 20).strokeBorder(Theme.ok, lineWidth: 4))
                            Text("New reference · camera frame from \(model.frameAge.map { Format.duration($0) } ?? "?") ago")
                                .font(.ui(28)).foregroundStyle(Theme.muted)
                        }
                    }
                    VStack(alignment: .leading, spacing: 28) {
                        Text(dialog == .disarm
                             ? "While Armed is off, no notification is sent to this TV or to the phones. The camera and history keep running."
                             : "The camera's current picture becomes the example of the correct position. Every position check from now on is compared with it.")
                            .font(.ui(40))
                        Label(dialog == .disarm ? "Nobody will be told if something goes wrong." : "Only do this while the baby lies exactly the way you want.",
                              systemImage: "exclamationmark.triangle")
                            .font(.ui(34)).foregroundStyle(Theme.high)
                            .padding(24).frame(maxWidth: .infinity, alignment: .leading)
                            .background(RoundedRectangle(cornerRadius: 24).fill(Theme.high.opacity(0.12)))
                            .overlay(RoundedRectangle(cornerRadius: 24).strokeBorder(Theme.high.opacity(0.55), lineWidth: 3))
                    }
                }
                HStack(spacing: 32) {
                    Spacer()
                    Button { model.dialog = nil } label: { pill("Cancel", filled: true) }
                        .buttonStyle(FocusStyle(radius: 56)).focused($focus)
                    Button(role: .destructive) {
                        let d = dialog
                        model.dialog = nil
                        if d == .disarm { model.set(.armed, to: false) } else { Task { await model.confirmReference() } }
                    } label: { pill(dialog == .disarm ? "Turn off" : "Use this frame", filled: false) }
                        .buttonStyle(FocusStyle(radius: 56))
                }
            }
            .foregroundStyle(Theme.text)
            .padding(.horizontal, 72).padding(.vertical, 60)
            .frame(width: 1500)
            .background(RoundedRectangle(cornerRadius: 44).fill(Color(hex: 0x1A1814)))
        }
        .focusSection()
        .onAppear { focus = true }
        .onExitCommand { model.dialog = nil }
    }

    private func pill(_ text: String, filled: Bool) -> some View {
        Text(text).font(.ui(44, bold: true)).padding(.horizontal, 64).frame(height: 112)
            .foregroundStyle(filled ? Color(hex: 0x17140F) : Theme.high)
            .background(Capsule().fill(filled ? Theme.text : .clear))
            .overlay(Capsule().strokeBorder(Theme.high, lineWidth: filled ? 0 : 4))
    }
}
