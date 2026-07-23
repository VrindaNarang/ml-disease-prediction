import { Box, Button, Typography } from '@mui/material'
import ArrowBackRoundedIcon from '@mui/icons-material/ArrowBackRounded'
import { useNavigate } from 'react-router-dom'

export default function NotFound() {
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
      <Typography variant="h1" fontWeight={800} color="primary.main" sx={{ fontSize: { xs: 64, sm: 96 } }}>
        404
      </Typography>
      <Typography variant="h6" fontWeight={600} sx={{ mt: 1 }}>
        Page not found
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1, mb: 3, maxWidth: 380 }}>
        The page you're looking for doesn't exist or may have moved.
      </Typography>
      <Button variant="contained" startIcon={<ArrowBackRoundedIcon />} onClick={() => navigate('/')}>
        Back to Dashboard
      </Button>
    </Box>
  )
}
