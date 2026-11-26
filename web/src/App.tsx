import type { ReactNode } from 'react'
import { createBrowserRouter, Navigate, RouterProvider } from 'react-router'
import { useAuth } from '@/features/auth/auth'
import { LoginPage } from '@/features/auth/LoginPage'
import { ChatView } from '@/features/chat/ChatView'
import { AppShell } from '@/features/shell/AppShell'

function RequireAuth({ children }: { children: ReactNode }) {
  const { token } = useAuth()
  return token ? children : <Navigate to="/login" replace />
}

const router = createBrowserRouter([
  { path: '/login', element: <LoginPage /> },
  {
    path: '/',
    element: (
      <RequireAuth>
        <AppShell />
      </RequireAuth>
    ),
    children: [
      { index: true, element: <ChatView /> },
      { path: 'c/:convId', element: <ChatView /> },
    ],
  },
  { path: '*', element: <Navigate to="/" replace /> },
])

export default function App() {
  return <RouterProvider router={router} />
}
