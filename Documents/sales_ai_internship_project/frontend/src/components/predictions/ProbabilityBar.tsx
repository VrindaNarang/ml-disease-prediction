import { Box, Typography } from '@mui/material'
import { useTheme } from '@mui/material/styles'

interface ProbabilityBarProps {
  value: number // 0-100
  width?: number
}

export function ProbabilityBar({ value, width = 90 }: ProbabilityBarProps) {
  const theme = useTheme()
  const color =
    value >= 70 ? theme.palette.success.main : value >= 40 ? theme.palette.warning.main : theme.palette.error.main

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
            width: `${Math.max(0, Math.min(100, value))}%`,
            height: '100%',
            bgcolor: color,
            borderRadius: 3,
          }}
        />
      </Box>
      <Typography variant="body2" fontWeight={700} sx={{ minWidth: 40, color }}>
        {value.toFixed(0)}%
      </Typography>
    </Box>
  )
}
