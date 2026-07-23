import { useEffect, useRef, useState } from 'react'
import {
  Alert,
  Box,
  Chip,
  IconButton,
  Paper,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material'
import ExpandMoreRoundedIcon from '@mui/icons-material/ExpandMoreRounded'
import ExpandLessRoundedIcon from '@mui/icons-material/ExpandLessRounded'
import SendRoundedIcon from '@mui/icons-material/SendRounded'
import SmartToyOutlinedIcon from '@mui/icons-material/SmartToyOutlined'
import PersonOutlineIcon from '@mui/icons-material/PersonOutline'
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded'
import DeleteOutlineRoundedIcon from '@mui/icons-material/DeleteOutlineRounded'
import AutoAwesomeRoundedIcon from '@mui/icons-material/AutoAwesomeRounded'
import { useCopilotChat } from '../../hooks/useCopilotChat'

const SUGGESTED_QUESTIONS = [
  'Why did sales decrease this month?',
  'Which customers should I contact this week?',
  'Which customers have the highest purchase probability?',
  'Which customers are at risk?',
  'Which region generated the highest revenue?',
  'Which recommendations are currently the highest priority?',
]

function TypingDots() {
  return (
    <Stack direction="row" spacing={0.6} sx={{ py: 0.5, px: 0.5 }}>
      {[0, 1, 2].map((i) => (
        <Box
          key={i}
          sx={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            bgcolor: 'text.secondary',
            animation: 'copilot-typing-dot 1.2s infinite ease-in-out',
            animationDelay: `${i * 0.15}s`,
            '@keyframes copilot-typing-dot': {
              '0%, 80%, 100%': { opacity: 0.2 },
              '40%': { opacity: 1 },
            },
          }}
        />
      ))}
    </Stack>
  )
}

export function CopilotPanel() {
  const [expanded, setExpanded] = useState(true)
  const [input, setInput] = useState('')
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)
  const { messages, isStreaming, error, sendMessage, clear } = useCopilotChat()
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (expanded) bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, expanded])

  const handleSend = (question?: string) => {
    const text = question ?? input
    if (!text.trim()) return
    void sendMessage(text)
    setInput('')
  }

  const handleCopy = (content: string, index: number) => {
    void navigator.clipboard.writeText(content)
    setCopiedIndex(index)
    setTimeout(() => setCopiedIndex((current) => (current === index ? null : current)), 1500)
  }

  return (
    <Paper variant="outlined" sx={{ mt: 4 }}>
      <Stack
        direction="row"
        alignItems="center"
        justifyContent="space-between"
        sx={{ px: 3, py: 2, cursor: 'pointer' }}
        onClick={() => setExpanded((v) => !v)}
      >
        <Stack direction="row" alignItems="center" spacing={1.5}>
          <AutoAwesomeRoundedIcon color="primary" />
          <Box>
            <Typography variant="subtitle1" fontWeight={700}>
              Sales Copilot
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Ask a business question — every answer is grounded in live analytics, predictions, and
              recommendations.
            </Typography>
          </Box>
        </Stack>
        <IconButton
          size="small"
          onClick={(e) => {
            e.stopPropagation()
            setExpanded((v) => !v)
          }}
          aria-label={expanded ? 'Collapse Sales Copilot panel' : 'Expand Sales Copilot panel'}
          aria-expanded={expanded}
        >
          {expanded ? <ExpandLessRoundedIcon /> : <ExpandMoreRoundedIcon />}
        </IconButton>
      </Stack>

      {expanded && (
        <Box sx={{ px: 3, pb: 3 }}>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 2 }}>
            {SUGGESTED_QUESTIONS.map((q) => (
              <Chip
                key={q}
                label={q}
                size="small"
                variant="outlined"
                clickable
                disabled={isStreaming}
                onClick={() => handleSend(q)}
              />
            ))}
          </Stack>

          <Box
            sx={{
              height: 420,
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 1.5,
              p: 2,
              mb: 2,
              borderRadius: 1,
              bgcolor: 'background.default',
              border: '1px solid',
              borderColor: 'divider',
            }}
          >
            {messages.length === 0 && (
              <Typography variant="body2" color="text.secondary" sx={{ m: 'auto', textAlign: 'center' }}>
                Ask a question above or pick a suggestion to get started.
              </Typography>
            )}
            {messages.map((msg, idx) => {
              const isLastAssistant = msg.role === 'assistant' && idx === messages.length - 1
              const showTyping = isLastAssistant && isStreaming && !msg.content
              return (
                <Box
                  key={idx}
                  sx={{
                    display: 'flex',
                    gap: 1,
                    alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                    flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                    maxWidth: '85%',
                  }}
                >
                  {msg.role === 'assistant' ? (
                    <SmartToyOutlinedIcon fontSize="small" color="primary" sx={{ mt: 0.5, flexShrink: 0 }} />
                  ) : (
                    <PersonOutlineIcon fontSize="small" sx={{ mt: 0.5, flexShrink: 0 }} />
                  )}
                  <Paper
                    variant="outlined"
                    sx={{
                      px: 1.5,
                      py: 1,
                      minWidth: showTyping ? 48 : undefined,
                      backgroundColor: msg.role === 'user' ? 'primary.main' : 'background.paper',
                      color: msg.role === 'user' ? 'primary.contrastText' : 'text.primary',
                      position: 'relative',
                    }}
                  >
                    {showTyping ? (
                      <TypingDots />
                    ) : (
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', pr: msg.role === 'assistant' ? 2.5 : 0 }}>
                        {msg.content}
                      </Typography>
                    )}
                    {msg.role === 'assistant' && msg.content && (
                      <Tooltip title={copiedIndex === idx ? 'Copied!' : 'Copy response'}>
                        <IconButton
                          size="small"
                          onClick={() => handleCopy(msg.content, idx)}
                          sx={{ position: 'absolute', top: 2, right: 2, opacity: 0.6 }}
                          aria-label="Copy response to clipboard"
                        >
                          <ContentCopyRoundedIcon sx={{ fontSize: 14 }} />
                        </IconButton>
                      </Tooltip>
                    )}
                  </Paper>
                </Box>
              )
            })}
            <div ref={bottomRef} />
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Stack direction="row" spacing={1}>
            <TextField
              fullWidth
              size="small"
              placeholder="Ask about sales, customers, predictions, or recommendations…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSend()
                }
              }}
              disabled={isStreaming}
              slotProps={{ htmlInput: { 'aria-label': 'Ask the Sales Copilot a question' } }}
            />
            <IconButton
              color="primary"
              onClick={() => handleSend()}
              disabled={isStreaming || !input.trim()}
              aria-label="Send message"
            >
              <SendRoundedIcon />
            </IconButton>
            <Tooltip title="Clear conversation">
              <span>
                <IconButton onClick={clear} disabled={isStreaming || messages.length === 0} aria-label="Clear conversation">
                  <DeleteOutlineRoundedIcon />
                </IconButton>
              </span>
            </Tooltip>
          </Stack>
        </Box>
      )}
    </Paper>
  )
}
