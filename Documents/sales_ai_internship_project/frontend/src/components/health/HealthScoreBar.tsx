import { Box, Typography } from '@mui/material'
import { useTheme } from '@mui/material/styles'
import type { HealthStatus } from '../../api/types'

interface HealthScoreBarProps {
  score: number
  status: HealthStatus
  width?: number
}

const STATUS_COLOR_KEY: Record<HealthStatus, 'success' | 'warning' | 'error'> = {
  healthy: 'success',
  at_risk: 'warning',
  critical: 'error',
}

export function HealthScoreBar({ score, status, width = 90 }: HealthScoreBarProps) {
  const theme = useTheme()
  const color = theme.palette[STATUS_COLOR_KEY[status]].main

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Box
        sx={{
          width,
          height: 6,
          borderRadius: 3,
          bgcolor: theme.palette.action.hover,
          overflow: 'hidden',
          flexShrink: 0,
        }}
      >
        <Box
          sx={{
            width: `${Math.max(0, Math.min(100, score))}%`,
            height: '100%',
            bgcolor: color,
            borderRadius: 3,
          }}
        />
      </Box>
      <Typography variant="body2" fontWeight={700} sx={{ minWidth: 32, color }}>
        {score.toFixed(0)}
      </Typography>
    </Box>
  )
}
