import SwiftUI

@main
struct BabymonApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView().environment(model)
        }
    }
}
