Get-ChildItem -Directory | ForEach-Object {
    $folderName = $_.Name
    $imagesPath = Join-Path $_.FullName "images"

    Write-Host "Procesando carpeta: ${imagesPath}" -ForegroundColor Yellow

    if (Test-Path $imagesPath) {
        # Lista todos los archivos en la carpeta "imagenes"
        $files = Get-ChildItem -Path $imagesPath -File

        Write-Host "Archivos encontrados en ${imagesPath}:" -ForegroundColor Cyan
        $files | ForEach-Object { Write-Host $_.Name }

        # Filtra los archivos que coinciden con el patrón
        $matchingFiles = $files | Where-Object {
            # Depuración: imprime cada nombre de archivo y el resultado de la expresión regular
            Write-Host "Analizando: $($_.Name)" -ForegroundColor Magenta
            $_.Name -match "^${folderName}_(\d+)\.JPEG$" -and [int]($matches[1]) -ge 50 -and [int]($matches[1]) -le 499
        }

        Write-Host "Archivos coincidentes:" -ForegroundColor Green
        $matchingFiles | ForEach-Object { Write-Host $_.Name }
    } else {
        Write-Host "No se encontró la carpeta: ${imagesPath}" -ForegroundColor Red
    }
}
