Get-ChildItem -Directory | ForEach-Object {
    $folderName = $_.Name
    $imagesPath = Join-Path $_.FullName "images"

    # Muestra las rutas que está procesando
    Write-Host "Procesando carpeta: $imagesPath" -ForegroundColor Yellow

    if (Test-Path $imagesPath) {
        # Lista todos los archivos en la carpeta "images"
        $files = Get-ChildItem -Path $imagesPath -File

        Write-Host "Archivos encontrados en ${imagesPath}:" -ForegroundColor Cyan
        $files | ForEach-Object { Write-Host $_.Name }

        # Filtra los archivos que coinciden con el patrón
        $matchingFiles = $files | Where-Object {
            $_.Name -match "^$folderName_([5-9][0-9]|[1-4][0-9]{2})\.JPEG$"
        }

        Write-Host "Archivos coincidentes:" -ForegroundColor Green
        $matchingFiles | ForEach-Object { Write-Host $_.Name }
    } else {
        Write-Host "No se encontró la carpeta: $imagesPath" -ForegroundColor Red
    }
}