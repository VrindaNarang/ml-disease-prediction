import { Box, Chip, IconButton, Paper, Skeleton, Stack, Typography } from '@mui/material'
import ArrowBackRoundedIcon from '@mui/icons-material/ArrowBackRounded'
import { useNavigate, useParams } from 'react-router-dom'
import { KpiCard } from '../components/dashboard/KpiCard'
import { ChartCard } from '../components/charts/ChartCard'
import { ProductTrendChart } from '../components/charts/ProductTrendChart'
import { ErrorState } from '../components/common/ErrorState'
import { useProduct, useProductSalesTrend } from '../hooks/useProductQueries'
import { formatCurrency, formatNumber } from '../utils/format'

export default function ProductDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const product = useProduct(id)
  const trend = useProductSalesTrend(id, 12)

  if (product.isError) {
    return <ErrorState message="Product not found or unreachable." onRetry={() => product.refetch()} />
  }

  const p = product.data
  const margin = p ? p.unit_price - p.cost_price : 0
  const marginPct = p && p.unit_price > 0 ? (margin / p.unit_price) * 100 : 0

  return (
    <>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <IconButton onClick={() => navigate('/products')} size="small" aria-label="Back to products">
          <ArrowBackRoundedIcon fontSize="small" />
        </IconButton>
        <Typography variant="body2" color="text.secondary">
          Back to Products
        </Typography>
      </Box>

      <Paper variant="outlined" sx={{ p: 3, mb: 3 }}>
        {product.isLoading || !p ? (
          <Stack spacing={1}>
            <Skeleton variant="text" width="40%" height={36} />
            <Skeleton variant="text" width="30%" />
          </Stack>
        ) : (
          <Box
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', md: 'row' },
              justifyContent: 'space-between',
              gap: 2,
            }}
          >
            <Box>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                <Typography variant="h5" fontWeight={700}>
                  {p.name}
                </Typography>
                <Chip
                  size="small"
                  label={p.is_active ? 'Active' : 'Inactive'}
                  color={p.is_active ? 'success' : 'default'}
                />
              </Stack>
              <Typography variant="body2" color="text.secondary">
                SKU {p.sku} · {p.category} · sold per {p.unit_of_measure}
              </Typography>
            </Box>
            <Stack spacing={0.5} sx={{ textAlign: { xs: 'left', md: 'right' } }}>
              <Typography variant="h5" fontWeight={700}>
                {formatCurrency(p.unit_price)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Cost {formatCurrency(p.cost_price)} · Margin {marginPct.toFixed(0)}%
              </Typography>
            </Stack>
          </Box>
        )}
      </Paper>

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: 'repeat(3, 1fr)' },
          gap: 2,
          mb: 3,
        }}
      >
        <KpiCard
          label="Total Revenue"
          value={p ? formatCurrency(p.total_revenue) : undefined}
          loading={product.isLoading}
        />
        <KpiCard
          label="Units Sold"
          value={p ? formatNumber(p.total_quantity_sold) : undefined}
          loading={product.isLoading}
        />
        <KpiCard
          label="Orders Containing This Product"
          value={p ? formatNumber(p.total_orders) : undefined}
          loading={product.isLoading}
        />
      </Box>

      <ChartCard
        title="Sales Trend"
        subtitle="Revenue and units sold over the last 12 months"
        loading={trend.isLoading}
        error={trend.isError}
        onRetry={() => trend.refetch()}
        isEmpty={(trend.data ?? []).length === 0}
      >
        <ProductTrendChart data={trend.data ?? []} />
      </ChartCard>
    </>
  )
}
