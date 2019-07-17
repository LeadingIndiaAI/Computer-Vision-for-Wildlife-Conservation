$files = ls *.xml
foreach ($file in $files){
[xml]$contents = gc $file.fullname
$xmlelement_file = $contents.CreateElement('Order')
$xmlelement_file.Innertext=$file.basename+'.jpg'
$contents.DocumentElement.AppendChild($xmlelement_file)
$contents.Save($file.fullname)
}