import { Outlet } from "react-router-dom"

import { AppSidebar } from "./AppSidebar"

export function AppShell() {
  return (
    <div className="flex h-screen overflow-hidden">
      <AppSidebar />
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}
