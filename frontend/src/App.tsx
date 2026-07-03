import { Navigate, Route, Routes } from "react-router-dom"

import { AppShell } from "@/components/layout/AppShell"
import { ComparePage } from "@/pages/ComparePage"
import { NewRunPage } from "@/pages/NewRunPage"
import { RunDetailPage } from "@/pages/RunDetailPage"
import { RunsPage } from "@/pages/RunsPage"

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/runs" replace />} />
        <Route path="/new" element={<NewRunPage />} />
        <Route path="/runs" element={<RunsPage />} />
        <Route path="/runs/:runId/*" element={<RunDetailPage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="*" element={<Navigate to="/runs" replace />} />
      </Route>
    </Routes>
  )
}
