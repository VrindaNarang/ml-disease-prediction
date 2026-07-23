import { Box, Button, Typography } from '@mui/material'
import LockOutlinedIcon from '@mui/icons-material/LockOutlined'
import ArrowBackRoundedIcon from '@mui/icons-material/ArrowBackRounded'
import { useNavigate } from 'react-router-dom'

export default function Forbidden() {
  const navigate = useNavigate()
  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        p: 3,
      }}
    >
      <LockOutlinedIcon color="error" sx={{ fontSize: 64, mb: 1 }} />
      <Typography variant="h6" fontWeight={600}>
        Access restricted
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1, mb: 3, maxWidth: 380 }}>
        Your account role doesn't have permission to view this page. Contact an administrator if you
        believe this is a mistake.
      </Typography>
      <Button variant="contained" startIcon={<ArrowBackRoundedIcon />} onClick={() => navigate('/')}>
        Back to Dashboard
      </Button>
    </Box>
  )
}
