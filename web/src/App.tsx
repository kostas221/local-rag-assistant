import { lazy, Suspense, type ReactNode } from 'react'
import { createBrowserRouter, Navigate, RouterProvider } from 'react-router'
import { ErrorPage } from '@/components/ErrorPage'
import { useAuth } from '@/features/auth/auth'

// Χωριστά κομμάτια κώδικα: η σελίδα σύνδεσης δεν φορτώνει markdown, chat και πάνελ σελίδας (Φάση 3).
const LoginPage = lazy(() => import('@/features/auth/LoginPage').then((m) => ({ default: m.LoginPage })))
const AppShell = lazy(() => import('@/features/shell/AppShell').then((m) => ({ default: m.AppShell })))
const ChatView = lazy(() => import('@/features/chat/ChatView').then((m) => ({ default: m.ChatView })))

function RequireAuth({ children }: { children: ReactNode }) {
  const { token } = useAuth()
  return token ? children : <Navigate to="/login" replace />
}

/** Όσο φορτώνει ένα κομμάτι κώδικα: σκέτο φόντο, χωρίς αναβόσβημα. */
function Splash() {
  return <div className="h-svh bg-background" aria-busy="true" />
}

const router = createBrowserRouter([
  { path: '/login', element: <LoginPage />, errorElement: <ErrorPage /> },
  {
    path: '/',
    element: (
      <RequireAuth>
        <AppShell />
      </RequireAuth>
    ),
    errorElement: <ErrorPage />,
    children: [
      { index: true, element: <ChatView /> },
      { path: 'c/:convId', element: <ChatView /> },
    ],
  },
  { path: '*', element: <Navigate to="/" replace /> },
])

export default function App() {
  return (
    <Suspense fallback={<Splash />}>
      <RouterProvider router={router} />
    </Suspense>
  )
}
