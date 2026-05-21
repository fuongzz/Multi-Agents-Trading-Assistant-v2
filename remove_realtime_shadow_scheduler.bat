@echo off
chcp 65001 > nul

schtasks /Delete /TN "AI-Trading-Shadow-Morning" /F
schtasks /Delete /TN "AI-Trading-Shadow-Afternoon" /F
