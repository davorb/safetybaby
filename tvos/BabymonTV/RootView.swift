import SwiftUI

struct RootView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.scenePhase) private var phase

    var body: some View {
        ZStack {
            Theme.ground.ignoresSafeArea()
            if model.serverURL == nil {
                SetupView()
            } else {
                switch model.screen {
                case .monitor: MonitorView()
                case .history: HistoryView()
                case .settings: SettingsView()
                }
                // The alert layer sits above every screen, including History and Settings.
                if let e = model.emergency {
                    if model.emergencyFullScreen {
                        EmergencyView(alert: e).transition(.opacity)
                    } else {
                        EmergencyBar(alert: e).frame(maxHeight: .infinity, alignment: .top)
                    }
                }
                if let d = model.dialog { DialogView(dialog: d) }
            }
        }
        .preferredColorScheme(.dark)
        .task(id: model.serverURL) { if model.serverURL != nil { model.restart() } }
        .onChange(of: phase) { _, new in model.scenePhaseChanged(active: new == .active) }
        .onChange(of: model.screen, initial: true) { _, s in UIApplication.shared.isIdleTimerDisabled = s == .monitor }
    }
}
