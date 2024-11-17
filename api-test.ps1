$headers = @{
    "Content-Type" = "application/json"
}

$body = @{
    "features" = @(0.5721456332346866, $false, $false, $false, $false, $false, $true, $false, $false, $false, $false, $false, $true, $false, $false, $false, $false, $true, $false, $false, $true, $true, $true, $false, $false, $true, $false, $false, $false, $false, $false, $true, $false, $false, $false, $false, $false, $false, $true, $false)
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Headers $headers -Body $body
