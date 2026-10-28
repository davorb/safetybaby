import SwiftUI

struct EmergencyView: View {
    let alert: ActiveAlert
    @Environment(AppModel.self) private var model
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @FocusState private var focus: Field?
    @State private var breathe = false
    @State private var holding = false

    enum Field { case back, ack }

    var body: some View {
        ZStack(alignment: .topLeading) {
            Theme.emergencyGround.ignoresSafeArea()
            RadialGradient(colors: [Theme.emergencyEdge.opacity(0.18), .clear], center: .init(x: 0.25, y: 0.4), startRadius: 0, endRadius: 800)
                .ignoresSafeArea()
            Rectangle().strokeBorder(Theme.emergencyEdge, lineWidth: reduceMotion ? 12 : 16)
                .opacity(reduceMotion ? 1 : (breathe ? 0.55 : 1)).ignoresSafeArea()

            HStack(alignment: .top, spacing: 60) {
                VStack(alignment: .leading, spacing: 0) {
                    Label("EMERGENCY", systemImage: "exclamationmark.triangle.fill")
                        .font(.ui(36, bold: true)).tracking(3).foregroundStyle(Color(hex: 0x2A0507))
                        .padding(.horizontal, 32).frame(height: 72)
                        .background(Capsule().fill(Theme.emergencyEdge))
                    Text(alert.title)
                        .font(.ui(150, bold: true)).lineLimit(2).minimumScaleFactor(0.5)
                        .padding(.top, 44)
                    Text(alert.message).font(.ui(48)).padding(.top, 36).lineLimit(3)
                    Text("\(alert.at.formatted(.dateTime.hour().minute().second())) · \(alert.delivered ? "Push sent to both phones" : "Push failed. The phones may not know.")")
                        .font(.ui(32)).foregroundStyle(Theme.emergencyMuted).padding(.top, 30)
                    Spacer(minLength: 24)
                    HStack(spacing: 28) {
                        Button { model.backToCamera() } label: {
                            Text("Back to camera").font(.ui(44, bold: true)).foregroundStyle(Color(hex: 0x3A0709))
                                .padding(.horizontal, 56).frame(height: 112).background(Capsule().fill(Theme.emergencyText))
                        }.buttonStyle(FocusStyle(radius: 56)).focused($focus, equals: .back)

                        Text(holding ? "Keep holding…" : "Hold Select to acknowledge")
                            .font(.ui(40, bold: true)).lineLimit(1).fixedSize()
                            .padding(.horizontal, 48).frame(height: 112)
                            .overlay(Capsule().strokeBorder(Theme.emergencyMuted, lineWidth: 4))
                            .modifier(FocusRing(focused: focus == .ack, radius: 56))
                            .focusable().focused($focus, equals: .ack)
                            .onLongPressGesture(minimumDuration: 1.5, perform: { model.clearEmergency() },
                                                onPressingChanged: { holding = $0 })
                    }
                    .focusSection()
                }
                .frame(width: 900, alignment: .leading)

                VStack(alignment: .leading, spacing: 28) {
                    Text("Snapshot at alert").font(.ui(30, bold: true)).foregroundStyle(Theme.emergencyMuted)
                    snapshot.frame(width: 780, height: 439).clipShape(RoundedRectangle(cornerRadius: 24))
                        .overlay(RoundedRectangle(cornerRadius: 24).strokeBorder(Theme.emergencyEdge, lineWidth: 6))
                    HStack(spacing: 24) {
                        Group {
                            if let f = model.frame { Image(uiImage: f).resizable().aspectRatio(contentMode: .fill) } else { Theme.panel }
                        }
                        .frame(width: 256, height: 144).clipShape(RoundedRectangle(cornerRadius: 14))
                        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Theme.emergencyMuted, lineWidth: 3))
                        VStack(alignment: .leading) {
                            Text("Live now").font(.ui(34, bold: true))
                            Text("Updated \(model.frameAge.map { Format.duration($0) } ?? "?") ago")
                                .font(.ui(30)).foregroundStyle(Theme.emergencyMuted)
                        }
                    }
                    Text("Sound repeats every 8 s\nPlay/Pause mutes for 60 s")
                        .font(.ui(30)).foregroundStyle(Theme.emergencyMuted)
                }
            }
            .foregroundStyle(Theme.emergencyText)
            .padding(.leading, 100).padding(.top, 70).padding(.bottom, 70)
        }
        .onAppear {
            focus = .back
            if !reduceMotion { withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) { breathe = true } }
        }
        .onPlayPauseCommand { model.muteEmergency() }
        .onExitCommand { model.backToCamera() }
    }

    @ViewBuilder private var snapshot: some View {
        if let url = alert.snapshotURL {
            AsyncImage(url: url) { $0.resizable().aspectRatio(contentMode: .fill) } placeholder: { Theme.panel }
        } else if let f = model.frame {
            Image(uiImage: f).resizable().aspectRatio(contentMode: .fill)
        } else { Theme.panel }
    }
}

/// What remains after Menu on the takeover: a red bar until the condition clears or it is acknowledged.
struct EmergencyBar: View {
    let alert: ActiveAlert
    @Environment(AppModel.self) private var model

    var body: some View {
        Button { model.emergencyFullScreen = true } label: {
            HStack(spacing: 22) {
                Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 40))
                Text("EMERGENCY · \(alert.title)").font(.ui(40, bold: true))
                Spacer()
                Text("Select to reopen").font(.ui(30))
            }
            .foregroundStyle(Theme.emergencyText)
            .padding(.horizontal, 40).frame(height: 96)
            .background(RoundedRectangle(cornerRadius: 28).fill(Theme.emergencyGround))
            .overlay(RoundedRectangle(cornerRadius: 28).strokeBorder(Theme.emergencyEdge, lineWidth: 5))
        }
        .buttonStyle(FocusStyle(radius: 28))
        .padding(.horizontal, 20).padding(.top, 8)
    }
}
