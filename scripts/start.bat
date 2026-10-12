@echo off
REM Usage: scripts\start.bat worker | scheduler
if "%1"=="worker" (
  docker compose -f docker-compose.worker.yml up -d --build
  docker compose -f docker-compose.worker.yml ps
) else if "%1"=="scheduler" (
  docker compose -f docker-compose.scheduler.yml up -d --build
  docker compose -f docker-compose.scheduler.yml ps
) else (
  echo Usage: scripts\start.bat worker ^| scheduler
)
