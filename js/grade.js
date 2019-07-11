var grade = function() {
  $('.icon-option').click(function() {
    $('.option').animate({left: "0px"}, 400);
  });
  $('.icon-close').click(function() {
    $('.option').animate({left: "-600px"}, 400);
  });
  $('.back').click(function() {
    $('#main-block').load('option.html');
  });
  $('.answer').click(function() {
    //show anser
  });
};

$(document).ready(grade);
