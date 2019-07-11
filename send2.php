<?php

$record = $_POST['record'];

echo $record;
$date = date('Y_m_d_H_i_s');
$myfile = fopen("log_".$date .".txt", "w") or die("Unable to open file!");
fwrite($myfile, $record);
fclose($myfile);

echo "<p><a href=\"http://localhost/index4.html\">回上一頁</a></p>";

?>