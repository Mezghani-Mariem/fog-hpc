# Envoie N jobs (en parallele) a un worker pour alimenter Grafana.
# Usage: .\scripts\send_jobs.ps1 -Url http://localhost:8001 -Count 20
param([string]$Url = "http://localhost:8001", [int]$Count = 20)
$body = '{"kind":"monte_carlo","size":100000,"seed":42}'
1..$Count | ForEach-Object -Parallel {
    Invoke-RestMethod -Method Post -Uri "$using:Url/jobs" -ContentType "application/json" -Body $using:body
} -ThrottleLimit 4 | Select-Object node_id, execution_seconds
