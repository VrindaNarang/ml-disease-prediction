import { Box, Card, CardContent, Skeleton, Typography } from '@mui/material'
import type { ReactNode } from 'react'

interface KpiCardProps {
  label: string
  value?: string
  icon?: ReactNode
  trendPct?: number
  loading?: boolean
}

export function KpiCard({ label, value, icon, trendPct, loading }: KpiCardProps) {
  const trendColor = trendPct === undefined ? undefined : trendPct >= 0 ? 'success.main' : 'error.main'

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <Typography variant="body2" color="text.secondary" fontWeight={500}>
            {label}
          </Typography>
          {icon && (
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 32,
                height: 32,
                borderRadius: 2,
                bgcolor: 'primary.main',
                color: 'primary.contrastText',
                opacity: 0.9,
              }}
            >
              {icon}
            </Box>
          )}
        </Box>

        {loading ? (
          <Skeleton variant="text" width="70%" height={36} sx={{ mt: 0.5 }} />
        ) : (
          <Typography variant="h5" fontWeight={700} sx={{ mt: 0.5 }}>
            {value}
          </Typography>
        )}

        {trendPct !== undefined && !loading && (
          <Typography variant="caption" sx={{ color: trendColor, fontWeight: 600 }}>
            {trendPct >= 0 ? '+' : ''}
            {trendPct.toFixed(1)}% vs last month
          </Typography>
        )}
      </CardContent>
    </Card>
  )
}
