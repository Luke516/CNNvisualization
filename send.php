<?php

$record = $_POST['record'];

echo $record;
$date = date('Y_m_d_H_i_s');
$myfile = fopen("log_".$date .".txt", "w") or die("Unable to open file!");
fwrite($myfile, $record);
fclose($myfile);


$addition = $_POST['addition'];

if(!empty($addition)){
	$myfile2 = fopen("addition.html", "w") or die("Unable to open file!");
	fwrite($myfile2, $addition);
	fclose($myfile2);
}

echo "<p><a href=\"http://localhost\">回上一頁</a></p>";

?>