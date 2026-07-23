import {
  Box,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  useMediaQuery,
} from '@mui/material'
import { useTheme } from '@mui/material/styles'
import DashboardOutlinedIcon from '@mui/icons-material/DashboardOutlined'
import PeopleAltOutlinedIcon from '@mui/icons-material/PeopleAltOutlined'
import Inventory2OutlinedIcon from '@mui/icons-material/Inventory2Outlined'
import InsightsOutlinedIcon from '@mui/icons-material/InsightsOutlined'
import TrendingUpRoundedIcon from '@mui/icons-material/TrendingUpRounded'
import RecommendRoundedIcon from '@mui/icons-material/RecommendRounded'
import SettingsOutlinedIcon from '@mui/icons-material/SettingsOutlined'
import ScienceOutlinedIcon from '@mui/icons-material/ScienceOutlined'
import { useLocation, useNavigate } from 'react-router-dom'

export const SIDEBAR_WIDTH = 240

const NAV_ITEMS = [
  { label: 'Dashboard', path: '/', icon: <DashboardOutlinedIcon /> },
  { label: 'Customers', path: '/customers', icon: <PeopleAltOutlinedIcon /> },
  { label: 'Products', path: '/products', icon: <Inventory2OutlinedIcon /> },
  { label: 'Analytics', path: '/analytics', icon: <InsightsOutlinedIcon /> },
  { label: 'Sales Opportunities', path: '/sales-opportunities', icon: <TrendingUpRoundedIcon /> },
  { label: 'Recommendations', path: '/recommendations', icon: <RecommendRoundedIcon /> },
  { label: 'Settings', path: '/settings', icon: <SettingsOutlinedIcon /> },
]

function isActive(pathname: string, itemPath: string) {
  if (itemPath === '/') return pathname === '/'
  return pathname === itemPath || pathname.startsWith(`${itemPath}/`)
}

interface SidebarProps {
  mobileOpen: boolean
  onClose: () => void
}

export function Sidebar({ mobileOpen, onClose }: SidebarProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const theme = useTheme()
  const isDesktop = useMediaQuery(theme.breakpoints.up('md'))

  const content = (
    <>
      <Toolbar sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 2 }}>
        <ScienceOutlinedIcon color="primary" />
        <Typography variant="h6" noWrap fontWeight={700}>
          SalesPilot AI
        </Typography>
      </Toolbar>
      <Box sx={{ overflow: 'auto', px: 1, py: 1 }}>
        <List sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
          {NAV_ITEMS.map((item) => {
            const selected = isActive(location.pathname, item.path)
            return (
              <ListItemButton
                key={item.path}
                selected={selected}
                onClick={() => {
                  navigate(item.path)
                  if (!isDesktop) onClose()
                }}
                sx={{
                  borderRadius: 2,
                  '&.Mui-selected': {
                    backgroundColor: 'primary.main',
                    color: 'primary.contrastText',
                    '& .MuiListItemIcon-root': { color: 'primary.contrastText' },
                    '&:hover': { backgroundColor: 'primary.dark' },
                  },
                }}
              >
                <ListItemIcon sx={{ minWidth: 36, color: selected ? 'inherit' : 'text.secondary' }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.label}
                  primaryTypographyProps={{ fontSize: 14, fontWeight: selected ? 600 : 500 }}
                />
              </ListItemButton>
            )
          })}
        </List>
      </Box>
    </>
  )

  if (isDesktop) {
    return (
      <Drawer
        variant="permanent"
        sx={{
          width: SIDEBAR_WIDTH,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: {
            width: SIDEBAR_WIDTH,
            boxSizing: 'border-box',
            backgroundColor: 'background.paper',
          },
        }}
      >
        {content}
      </Drawer>
    )
  }

  return (
    <Drawer
      variant="temporary"
      open={mobileOpen}
      onClose={onClose}
      ModalProps={{ keepMounted: true }}
      sx={{
        [`& .MuiDrawer-paper`]: {
          width: SIDEBAR_WIDTH,
          boxSizing: 'border-box',
          backgroundColor: 'background.paper',
        },
      }}
    >
      {content}
    </Drawer>
  )
}
