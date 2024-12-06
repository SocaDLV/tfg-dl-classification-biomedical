Get-ChildItem -Directory | ForEach-Object {
    $folderName = $_.Name
    $imagesPath = Join-Path $_.FullName "images"

    Write-Host "Procesando carpeta: ${imagesPath}" -ForegroundColor Yellow

    if (Test-Path $imagesPath) {
        # Lista todos los archivos en la carpeta "images"
        $files = Get-ChildItem -Path $imagesPath -File

        # Filtra los archivos que coinciden con el patrón
        $matchingFiles = $files | Where-Object {
            $_.Name -match "^${folderName}_(\d+)\.JPEG$" -and [int]($matches[1]) -ge 50 -and [int]($matches[1]) -le 499
        }

        # Elimina los archivos coincidentes
        if ($matchingFiles) {
            Write-Host "Eliminando archivos coincidentes en ${imagesPath}:" -ForegroundColor Red
            $matchingFiles | ForEach-Object {
                Write-Host "Eliminando: $($_.FullName)" -ForegroundColor Red
                Remove-Item -Path $_.FullName
            }
        } else {
            Write-Host "No se encontraron archivos para eliminar en ${imagesPath}" -ForegroundColor Green
        }
    } else {
        Write-Host "No se encontró la carpeta: ${imagesPath}" -ForegroundColor Red
    }
}
