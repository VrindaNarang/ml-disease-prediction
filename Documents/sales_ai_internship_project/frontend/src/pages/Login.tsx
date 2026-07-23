import { useState } from 'react'
import type { FormEvent } from 'react'
import {
  Alert,
  Box,
  Button,
  IconButton,
  InputAdornment,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import ScienceOutlinedIcon from '@mui/icons-material/ScienceOutlined'
import VisibilityRoundedIcon from '@mui/icons-material/VisibilityRounded'
import VisibilityOffRoundedIcon from '@mui/icons-material/VisibilityOffRounded'
import { useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const DEMO_ACCOUNTS = [
  { role: 'Admin', email: 'admin@salespilot.example.com' },
  { role: 'Sales Manager', email: 'manager@salespilot.example.com' },
  { role: 'Sales Executive', email: 'executive@salespilot.example.com' },
]
const DEMO_PASSWORD = 'ChangeMe123!'

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const redirectTo = (location.state as { from?: string } | null)?.from ?? '/'

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      await login(email.trim(), password)
      navigate(redirectTo, { replace: true })
    } catch {
      setError('Incorrect email or password.')
    } finally {
      setSubmitting(false)
    }
  }

  const fillDemoAccount = (demoEmail: string) => {
    setEmail(demoEmail)
    setPassword(DEMO_PASSWORD)
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'background.default',
        p: 2,
      }}
    >
      <Paper
        component="form"
        onSubmit={handleSubmit}
        variant="outlined"
        sx={{ width: '100%', maxWidth: 400, p: 4 }}
      >
        <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 3 }}>
          <ScienceOutlinedIcon color="primary" fontSize="large" />
          <Box>
            <Typography variant="h6" fontWeight={700}>
              SalesPilot AI
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Sign in to your account
            </Typography>
          </Box>
        </Stack>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Stack spacing={2}>
          <TextField
            label="Email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoFocus
            fullWidth
            autoComplete="username"
            slotProps={{ htmlInput: { 'aria-label': 'Email address' } }}
          />
          <TextField
            label="Password"
            type={showPassword ? 'text' : 'password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            fullWidth
            autoComplete="current-password"
            slotProps={{
              htmlInput: { 'aria-label': 'Password' },
              input: {
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowPassword((v) => !v)}
                      edge="end"
                      size="small"
                      aria-label={showPassword ? 'Hide password' : 'Show password'}
                    >
                      {showPassword ? <VisibilityOffRoundedIcon fontSize="small" /> : <VisibilityRoundedIcon fontSize="small" />}
                    </IconButton>
                  </InputAdornment>
                ),
              },
            }}
          />
          <Button type="submit" variant="contained" size="large" disabled={submitting} fullWidth>
            {submitting ? 'Signing in…' : 'Sign In'}
          </Button>
        </Stack>

        <Box sx={{ mt: 3, pt: 2.5, borderTop: '1px solid', borderColor: 'divider' }}>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
            Demo accounts (password: {DEMO_PASSWORD}) — click to fill in:
          </Typography>
          <Stack spacing={0.5}>
            {DEMO_ACCOUNTS.map((account) => (
              <Button
                key={account.email}
                size="small"
                variant="text"
                onClick={() => fillDemoAccount(account.email)}
                sx={{ justifyContent: 'flex-start', textTransform: 'none', color: 'text.secondary' }}
              >
                <Typography variant="caption">
                  <strong>{account.role}</strong> — {account.email}
                </Typography>
              </Button>
            ))}
          </Stack>
        </Box>
      </Paper>
    </Box>
  )
}
