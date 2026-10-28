import SwiftUI

struct SettingsView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ZStack {
            Theme.ground.ignoresSafeArea()
            VStack(alignment: .leading, spacing: 36) {
                Text("Settings").font(.ui(72, bold: true))
                section("Server") {
                    Text(model.serverURL?.absoluteString ?? "Not set").font(.ui(40, bold: true))
                    action("Change server", "network") { model.forgetServer() }
                }
                section("Alert sound") {
                    Text("The app cannot change the TV volume. Play the tones to check the level.")
                        .font(.ui(30)).foregroundStyle(Theme.muted)
                    HStack(spacing: 24) {
                        action("Emergency alarm", "speaker.wave.3.fill") { model.playTest(.emergency) }
                        action("High chime", "bell.fill") { model.playTest(.high) }
                    }
                }
                section("Screen") {
                    Text("The overlay dims to 60 % after 10 minutes with no remote input and drifts 6 px every 5 minutes to protect the panel. The screen stays awake while Monitor is open.")
                        .font(.ui(30)).foregroundStyle(Theme.muted)
                }
                section("About") {
                    Text("babymon for Apple TV · this TV is a second alert surface. Phone push stays the safety net.")
                        .font(.ui(30)).foregroundStyle(Theme.muted)
                }
                Spacer()
            }
            .foregroundStyle(Theme.text)
            .padding(.horizontal, 20)
        }
        .onExitCommand { model.screen = .monitor; model.dockOpen = true }
    }

    private func section(_ title: String, @ViewBuilder _ content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title).font(.ui(30, bold: true)).foregroundStyle(Theme.muted)
            content()
        }
    }

    private func action(_ title: String, _ symbol: String, _ run: @escaping () -> Void) -> some View {
        Button(action: run) {
            Label(title, systemImage: symbol).font(.ui(36, bold: true))
                .padding(.horizontal, 40).frame(height: 96)
                .background(Capsule().fill(Theme.raised))
        }.buttonStyle(FocusStyle(radius: 48))
    }
}
