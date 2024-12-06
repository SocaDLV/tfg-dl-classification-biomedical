# Recorre todas las carpetas dentro del directorio actual (tinyImgNet-train)
Get-ChildItem -Directory | ForEach-Object {
    $folderName = $_.Name  # Nombre de la carpeta (como n12345)
    $imagesPath = Join-Path $_.FullName "images"  # Ruta completa a la subcarpeta "images"

    # Verifica si la subcarpeta "images" existe
    if (Test-Path $imagesPath) {
        # Realiza el borrado de las imágenes dentro de la carpeta "images"
        Get-ChildItem -Path $imagesPath -File | Where-Object { 
            $_.Name -match "^$folderName_([5-9][0-9]|[1-4][0-9]{2})\.JPEG$" 
        } | Remove-Item -WhatIf
    }
}
