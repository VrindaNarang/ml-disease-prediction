import { InputAdornment, TextField } from '@mui/material'
import SearchRoundedIcon from '@mui/icons-material/SearchRounded'
import { useEffect, useRef, useState } from 'react'

interface SearchInputProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  debounceMs?: number
  size?: 'small' | 'medium'
  fullWidth?: boolean
}

export function SearchInput({
  value,
  onChange,
  placeholder = 'Search…',
  debounceMs = 350,
  size = 'small',
  fullWidth,
}: SearchInputProps) {
  const [draft, setDraft] = useState(value)
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>()

  useEffect(() => {
    setDraft(value)
  }, [value])

  const handleChange = (next: string) => {
    setDraft(next)
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    timeoutRef.current = setTimeout(() => onChange(next), debounceMs)
  }

  return (
    <TextField
      value={draft}
      onChange={(e) => handleChange(e.target.value)}
      placeholder={placeholder}
      size={size}
      fullWidth={fullWidth}
      slotProps={{
        input: {
          startAdornment: (
            <InputAdornment position="start">
              <SearchRoundedIcon fontSize="small" sx={{ color: 'text.secondary' }} />
            </InputAdornment>
          ),
        },
        htmlInput: { 'aria-label': placeholder },
      }}
    />
  )
}
