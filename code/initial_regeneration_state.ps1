# Estado inicial de regeneracion de las piezas del articulo CAD (SolidWorks, via su interop oficial .NET).
# Para cada modelo: abre (solo lectura, silencioso), cuenta errores de regeneracion como QualiCAD
# (EditRebuild3 + GetWhatsWrongCount) ANTES de modificar nada, y guarda el resultado en un CSV.
# Si se interrumpe, al volver a correrlo continua donde quedo.
param(
  [string]$Zip     = (Join-Path $PSScriptRoot '..\Piezas_solidworks_paper.zip'),
  [string]$Lista   = (Join-Path $PSScriptRoot 'lista_modelos_352.csv'),
  [string]$Salida  = (Join-Path $PSScriptRoot '..\04_resultados\estado_inicial_regeneracion.csv'),
  [int]$Max        = 0
)
$ErrorActionPreference = 'Stop'

# --- localizar el interop oficial de SolidWorks ---
$raices = @('C:\Program Files\SOLIDWORKS Corp', 'C:\Program Files (x86)\SOLIDWORKS Corp')
$dll = $null
foreach ($r in $raices) {
  if (Test-Path $r) {
    $dll = Get-ChildItem -Path $r -Recurse -Filter 'SolidWorks.Interop.sldworks.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($dll) { break }
  }
}
if (-not $dll) { throw 'No encontre SolidWorks.Interop.sldworks.dll dentro de C:\Program Files\SOLIDWORKS Corp' }
Write-Host ('Interop: ' + $dll.FullName)
Add-Type -Path $dll.FullName

$cs = @'
using System;
using SolidWorks.Interop.sldworks;
public static class SwHelper {
  public static object Conectar() {
    Type t = Type.GetTypeFromProgID("SldWorks.Application");
    ISldWorks sw = (ISldWorks)Activator.CreateInstance(t);
    try { sw.Visible = true; } catch { }
    return sw;
  }
  static int Contar(ModelDoc2 d) {
    try { dynamic ext = d.Extension; return (int)ext.GetWhatsWrongCount(); } catch { return -1; }
  }
  public static object[] Medir(object swo, string ruta) {
    ISldWorks sw = (ISldWorks)swo;
    int e = 0, w = 0; string abre = "si"; string nota = "";
    int c0 = -1, c1 = -1, c2 = -1;
    try {
      ModelDoc2 d = sw.OpenDoc6(ruta, 1, 3, "", ref e, ref w);
      if (d == null) { abre = "no"; nota = "OpenDoc6 devolvio null"; }
      else {
        c0 = Contar(d);
        d.EditRebuild3();
        c1 = Contar(d);
        d.ForceRebuild3(false);
        c2 = Contar(d);
        sw.CloseDoc(d.GetTitle());
      }
    } catch (Exception ex) { abre = "error"; nota = ex.Message; }
    return new object[] { abre, e, w, c0, c1, c2, nota };
  }
}
'@
Add-Type -ReferencedAssemblies @($dll.FullName, 'Microsoft.CSharp', 'System.Core') -TypeDefinition $cs

$dest = Join-Path $env:TEMP 'piezas_paper_estado_inicial'
$carpeta = Join-Path $dest 'Piezas_solidworks_paper'
if (-not (Test-Path $carpeta)) {
  Write-Host 'Descomprimiendo piezas...'
  Expand-Archive -Path $Zip -DestinationPath $dest -Force
}

$modelos = Import-Csv -Path $Lista -Encoding UTF8
if ($Max -gt 0) { $modelos = $modelos | Select-Object -First $Max }

# si el CSV solo tiene filas de un intento fallido (ninguna con abre = si), se descarta
if (Test-Path $Salida) {
  $prev = @(Import-Csv -Path $Salida -Encoding UTF8)
  if (@($prev | Where-Object { $_.abre -eq 'si' }).Count -eq 0) { Remove-Item $Salida -Force }
}
$hechos = @()
if (Test-Path $Salida) { $hechos = @((Import-Csv -Path $Salida -Encoding UTF8).file_name) }

Write-Host 'Iniciando/conectando con SolidWorks (puede tardar un minuto)...'
$sw = [SwHelper]::Conectar()
$n = @($modelos).Count
$i = 0
foreach ($m in $modelos) {
  $i++
  if ($hechos -contains $m.file_name) { continue }
  $ruta = [string](Join-Path $carpeta $m.file_name)
  if (-not (Test-Path $ruta)) {
    $r = @('error', 0, 0, -1, -1, -1, 'archivo no encontrado en el zip')
  } else {
    $r = [SwHelper]::Medir($sw, $ruta)
  }
  $abre = $r[0]; $c1 = $r[4]
  $reg = ''
  if ($abre -eq 'si' -and $c1 -ne -1) { if ($c1 -eq 0) { $reg = 1 } else { $reg = 0 } }
  [pscustomobject]@{
    model_id = $m.model_id; file_name = $m.file_name; source = $m.source
    success_ratio = $m.success_ratio; abre = $abre
    codigo_apertura = $r[1]; avisos_apertura = $r[2]
    errores_al_abrir = $r[3]; errores_editrebuild = $r[4]; errores_forcerebuild = $r[5]
    regenera_inicial = $reg; nota = $r[6]
  } | Export-Csv -Path $Salida -Append -NoTypeInformation -Encoding UTF8
  Write-Host ("{0}/{1} {2} -> abre={3} errores={4} regenera_inicial={5} {6}" -f $i, $n, $m.file_name, $abre, $c1, $reg, $r[6])
}
Write-Host "Listo. Resultados en: $Salida"
