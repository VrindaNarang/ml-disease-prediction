import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import Login from './Login'
import { useAuth } from '../context/AuthContext'

vi.mock('../context/AuthContext', () => ({
  useAuth: vi.fn(),
}))

const mockedUseAuth = vi.mocked(useAuth)

function renderLogin() {
  return render(
    <MemoryRouter initialEntries={['/login']} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Login />
    </MemoryRouter>,
  )
}

describe('Login page', () => {
  it('fills in a demo account email/password when a demo account button is clicked', async () => {
    mockedUseAuth.mockReturnValue({ login: vi.fn(), logout: vi.fn(), user: null, status: 'unauthenticated' })
    renderLogin()

    await userEvent.click(screen.getByRole('button', { name: /Admin.*admin@salespilot\.example\.com/ }))

    expect(screen.getByRole('textbox', { name: 'Email address' })).toHaveValue('admin@salespilot.example.com')
  })

  it('calls login with the entered credentials on submit', async () => {
    const login = vi.fn().mockResolvedValue(undefined)
    mockedUseAuth.mockReturnValue({ login, logout: vi.fn(), user: null, status: 'unauthenticated' })
    renderLogin()

    await userEvent.type(screen.getByRole('textbox', { name: 'Email address' }), 'manager@salespilot.example.com')
    await userEvent.type(screen.getByLabelText('Password'), 'ChangeMe123!')
    await userEvent.click(screen.getByRole('button', { name: 'Sign In' }))

    expect(login).toHaveBeenCalledWith('manager@salespilot.example.com', 'ChangeMe123!')
  })

  it('shows an error message when login fails', async () => {
    const login = vi.fn().mockRejectedValue(new Error('bad credentials'))
    mockedUseAuth.mockReturnValue({ login, logout: vi.fn(), user: null, status: 'unauthenticated' })
    renderLogin()

    await userEvent.type(screen.getByRole('textbox', { name: 'Email address' }), 'nobody@example.com')
    await userEvent.type(screen.getByLabelText('Password'), 'wrong-password')
    await userEvent.click(screen.getByRole('button', { name: 'Sign In' }))

    expect(await screen.findByText('Incorrect email or password.')).toBeInTheDocument()
  })

  it('toggles password visibility', async () => {
    mockedUseAuth.mockReturnValue({ login: vi.fn(), logout: vi.fn(), user: null, status: 'unauthenticated' })
    renderLogin()

    const passwordField = screen.getByLabelText('Password') as HTMLInputElement
    expect(passwordField.type).toBe('password')

    await userEvent.click(screen.getByRole('button', { name: 'Show password' }))
    expect(passwordField.type).toBe('text')
  })
})
