// Display mode selection
var question_type = ["vertical", "horizontal", "horizontal2"];
var MAX_NUM = 1000;
var all_questions;

var q_setting = 2;
var option_setting;
var questions;
var current_question;

var selected_questions=[];
var current_no=0;
var avilible=0;

var right = 0;
var wrong = 0;

var Question = function(topic, description, id, option){
  var temp;
  // randomize (swap) options, 20 times might be enough
  var ans = 1;
  for(i=0; i<20; i++){
    var random_number = Math.floor(Math.random()*4);
    var random_number2 = Math.floor(Math.random()*4);
    if(ans == random_number+ 1){
      ans= random_number2 + 1;
    } else if(ans == random_number2 + 1){
      ans= random_number + 1;
    }
    temp = option[random_number];
    option[random_number] = option[random_number2];
    option[random_number2] =temp;
  }
  this.id = id;
  this.topic = topic;
  this.description = description;
  this.option = { 1:option[0], 2:option[1], 3:option[2], 4:option[3] };
  this.answer = ans;
  this.selected_options = [];
  this.selected_answer = 0;
  this.result = 0; // 0: not yet answer, 1: correct, 2: wrong, 3: correct after wrong try
}

function quest_init(){
  $("#question-placeholder").load("quest-" + question_type[q_setting] + ".html",layout_init);
}

function layout_init(){
  //alert('layout init !');
  update_layout();

  $('#option-button').click(function(){
    $('#popup-option').toggleClass('popup-active');
  });
  $('#popup-option').click(function(){
    $('#popup-option').toggleClass('popup-active');
  });
  $('.mode-select').click(function(){
    setTimeout(update_layout,50);
    q_setting = $(this).attr('data');
    update_layout();
    setTimeout(question_display,300);
    setTimeout(btn_event,300);
  });
  /* after layout has been initialized, find first question */
  setTimeout(btn_event,300);
  find_question();
}

function update_layout(){
  $("#question-placeholder").load("quest-" + question_type[q_setting] + ".html");
}
function btn_event() {
  $('#option1, #option2, #option3, #option4').click(function() {
    var pressed = $(this).attr('id')[6];
    if( current_question.result == 1 || current_question.result == 3 ) return; // Already correct
    console.log(current_question.answer);
    if( current_question.answer == pressed ){
      current_question.selected_answer = answer;
      if(current_question.result == 0){
        current_question.result = 1;
        right++;
        // Corrert, add point!
      }else{
        current_question.result = 3;
        // Corrert after retry
      }
    }else{
      for(var i in current_question.selected_options){
        if(current_question.selected_options[i] == pressed )return; // Pressed selected option
      }
      current_question.selected_options[current_question.selected_options.length] = pressed;
      if(current_question.result != 2){
        // Wrong!
        wrong++;
      }
      current_question.result = 2;
    }
    QuestionTag.updateButtonStatus(current_question.selected_answer, current_question.selected_options);
    updateScore();
  });
}

function updateScore(){
  $('#score').text(100/option_setting[1]*right);
  $('#right').text(right);
  $('#wrong').text(wrong);
}

function question_display(){
  //alert('question_display!');
  QuestionTag.display(current_question);
  QuestionTag.updateButtonStatus(current_question.selected_answer, current_question.selected_options);
}

var next_quest = function(){
  current_no++;

  /* Exceed question count */
  if(current_no >= option_setting[1]){
    $("#end-test").removeClass("btn-danger");
    $("#end-test").addClass("btn-success");
    current_no = 0;
    current_question = selected_questions[0];
  }

  /* need to find new question */
  else if(selected_questions.length <= current_no){
    find_question();
  }

  /* just switch to question that had been found */
  else{
    current_question = selected_questions[current_no];
  }
  question_display();
}

var prev_quest = function(){
  current_no--;
  if(current_no<0)current_no=0;
    current_question = selected_questions[current_no];
  question_display();
}

$('#question-prev').click(function() {
  prev_quest();
});

$('#question-next').click(function() {
  next_quest();
});

var find_question = function(){
  //alert('find question!');
  /* Load all question from JSON file */
  $.getJSON("all_questions.json", function(questions) {
    var random_number = getRandomQuestionID();

    /* Check if question category is satisfied */
    while(questions[random_number].category=="0" && option_setting[0]==1){
      random_number = getRandomQuestionID();//check category
    }
    while(questions[random_number].category=="1" && option_setting[0]==2){
      random_number = getRandomQuestionID();
    }
    //alert(questions[random_number].question+' QWQ');
    //alert(questions[random_number].id);
    current_question = new Question(
      questions[random_number].question,
      questions[random_number].ps,
      questions[random_number].id,
      [
        questions[random_number].answer,
        questions[random_number].option1,
        questions[random_number].option2,
        questions[random_number].option3
      ]
    );
    //alert(current_question.topic+' QWQ2');
    selected_questions[current_no]=current_question;
    question_display();
  });
}

function getRandomQuestionID(){
  var random_number;
  do{
    /* If repeated, try again */
    random_number=Math.floor(Math.random()*MAX_NUM);
  }while( check_repeat(random_number) );
    return random_number;
}

var check_repeat = function(random_number){
  var i;
  for(i=0; i<selected_questions.length; i++){
    if((selected_questions[i].id-1) == random_number){
      return true;
    }
  }
  return false;
}

$("#end-test").click(function() {
  console.log('End Test');
  $('#main-block').load('newGrade.html');
});

$(document).ready(quest_init);
