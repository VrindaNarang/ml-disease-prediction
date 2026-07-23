import { createTheme, type PaletteMode } from '@mui/material/styles'

// Clean SaaS-style theme: indigo primary, neutral surfaces, subtle borders
// instead of heavy shadows. Supports both light and dark palettes.
const getDesignTokens = (mode: PaletteMode) => ({
  palette: {
    mode,
    primary: {
      main: '#4F46E5',
      light: '#818CF8',
      dark: '#3730A3',
    },
    secondary: {
      main: '#0EA5E9',
    },
    success: { main: '#16A34A' },
    warning: { main: '#D97706' },
    error: { main: '#DC2626' },
    ...(mode === 'light'
      ? {
          background: {
            default: '#F7F8FA',
            paper: '#FFFFFF',
          },
          text: {
            primary: '#1A1D29',
            secondary: '#6B7280',
          },
          divider: '#E5E7EB',
        }
      : {
          background: {
            default: '#0F1115',
            paper: '#171A21',
          },
          text: {
            primary: '#F3F4F6',
            secondary: '#9CA3AF',
          },
          divider: '#2A2E37',
        }),
  },
  shape: {
    borderRadius: 10,
  },
  typography: {
    fontFamily: [
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      'Helvetica',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: { fontWeight: 700 },
    h2: { fontWeight: 700 },
    h3: { fontWeight: 600 },
    h4: { fontWeight: 600 },
    h5: { fontWeight: 600 },
    h6: { fontWeight: 600 },
  },
})

export function getAppTheme(mode: PaletteMode) {
  const tokens = getDesignTokens(mode)
  return createTheme({
    ...tokens,
    components: {
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            boxShadow: 'none',
            borderBottom: `1px solid ${tokens.palette.divider}`,
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            borderRight: `1px solid ${tokens.palette.divider}`,
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            fontWeight: 600,
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            border: `1px solid ${tokens.palette.divider}`,
            boxShadow: 'none',
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            borderColor: tokens.palette.divider,
          },
        },
      },
    },
  })
}
