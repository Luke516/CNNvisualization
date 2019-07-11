var grade_main = function() {
  setTimeout(grade_display,200);

  $('#restart').click(function() {
    $('#main-block').load('option.html');
    selected_questions = [];
  });
}

var grade_display = function(){
  $('#correct-table').text(right);
  $('#wrong-table').text(wrong);
  $('#score-table').text(100/option_setting[1]*right);

}

$(document).ready(grade_main);
