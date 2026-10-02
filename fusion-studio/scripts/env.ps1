$env:PATH = "$env:USERPROFILE\.cargo\bin;" +
            "C:\tools\llvm\bin;" +
            "C:\tools\opencv\opencv\build\x64\vc16\bin;" +
            "C:\Program Files\Prophesee\bin;" +
            "C:\Program Files\Teledyne\Spinnaker\bin64\vs2015;" + $env:PATH
if (-not $env:MV_HAL_PLUGIN_PATH) {
  $env:MV_HAL_PLUGIN_PATH = "C:\Program Files\Prophesee\lib\metavision\hal\plugins"
}
