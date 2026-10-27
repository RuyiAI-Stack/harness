import { createRoot } from 'react-dom/client'
import App from './App'
import { AccountGate } from './AccountGate'
import './styles.css'

createRoot(document.getElementById('root')!).render(<AccountGate><App /></AccountGate>)
