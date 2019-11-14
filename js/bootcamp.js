var demo_id = 1;
var max_demo_id = 3;
var cur_layer_name = "block1_conv1";

var content_id = 0;
var style_id = 0;

function change_style_transfer(content_offset, style_offset){
  content_id = content_id + content_offset;
  style_id = style_id + style_offset;
  content_id = (content_id+10) % 10;
  style_id = (style_id+10) % 10;
  $("#content-image").attr({ "src": "../images/contents/" + (content_id+1) + ".jpg" });
  $("#style-image").attr({ "src": "../images/styles/" + (style_id+1) + ".jpg" });
  $("#style-transfered-image").attr({ "src": "../images/demo_style_transfer/style_transfer_" + (content_id+1) + "_" + + (style_id+1) + ".png" });
}

function clearHighlighted(){
  $("#"+cur_layer_name).removeClass('btn-highlighted');
  hightlightVGG(cur_layer_name, 3);
}

function loadFeatureMaps(layer_name) {
  console.log("loadFeatureMaps");
  console.log(layer_name);

  var count = 0;
  $("#feature_maps > img").each(function() {
    let src_url = "../images/demo_imgs/feature_map" + demo_id +"/" + layer_name + "_" + count + ".png";
    $(this).attr({ "src": src_url });
    count ++;
  });
}

function loadFilters(layer_name) {
  console.log("loadFilters");
  console.log(layer_name);

  var count = 0;
  $("#filters > img").each(function() {
    let src_url = "../images/demo_imgs/filter/" + layer_name + "_" + count + ".png";
    $(this).attr({ "src": src_url });
    count ++;
  });
}

function hightlightVGG(layer_name, flag){
  if(layer_name[5] == cur_layer_name[5] && flag < 2){
    return;
  }
  var color = "";
  if(flag == 1){
    color = "hsla(200, 100%, 80%, 0.7)";
  }else if(flag == 0 || flag==3){
    color = "hsla(180, 100%, 50%, 0.7)";
  }else{
    color = "hsla(20, 100%, 50%, 0.7)";
  }
  $(".front" + layer_name[5]).css("background",   color);
  $(".back" + layer_name[5]).css("background", color);
  $(".right" + layer_name[5]).css("background", color);
  $(".left" + layer_name[5]).css("background", color);
}

function refresh1() {
  console.log("refresh");
  loadFilters(cur_layer_name);
  loadFeatureMaps(cur_layer_name);
  $("#demo1_original").attr({ "src": "../images/demo_imgs/" + demo_id + ".jpg" });
}

$( document ).ready((function($) {
  "use strict"; // Start of use strict

  // Smooth scrolling using jQuery easing
  $('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function() {
    if (location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') && location.hostname == this.hostname) {
      var target = $(this.hash);
      target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
      if (target.length) {
        $('html, body').animate({
          scrollTop: (target.offset().top - 48)
        }, 1000, "easeInOutExpo");
        return false;
      }
    }
  });

  // Closes responsive menu when a scroll trigger link is clicked
  $('.js-scroll-trigger').click(function() {
    $('.navbar-collapse').collapse('hide');
  });

  /*$(function(){
    $('.explore').click(function() {
      var cls = $(this).closest("section").next().offset().top;
      alert(cls);
  		$("html, body").animate({scrollTop: cls}, "slow");
    });
  });*/

  var pagePositon = 0,
    sectionsSeclector = 'section',
    $scrollItems = $(sectionsSeclector),
    offsetTolorence = 30,
    pageMaxPosition = $scrollItems.length - 1;

  //Map the sections:
  $scrollItems.each(function(index,ele) { $(ele).attr("debog",index).data("pos",index); });

  // Bind to scroll
  $(window).bind('scroll',upPos);

  //Move on click:
  $('.explore').click(function(e){
      if (pagePositon+1 <= pageMaxPosition) {
          pagePositon++;
          $('html, body').stop().animate({
                scrollTop: $scrollItems.eq(pagePositon).offset().top
          }, 300);
      }
  });

  //Update position func:
  function upPos(){
    var fromTop = $(this).scrollTop();
    var $cur = null;
      $scrollItems.each(function(index,ele){
          if ($(ele).offset().top < fromTop + offsetTolorence) $cur = $(ele);
      });
    if ($cur != null && pagePositon != $cur.data('pos')) {
        pagePositon = $cur.data('pos');
    }
  }

  // Activate scrollspy to add active class to navbar items on scroll
  $('body').scrollspy({
    target: '#mainNav',
    offset: 54
  });

  // Collapse Navbar
  var navbarCollapse = function() {
    if ($("#mainNav").offset().top > 100) {
      $("#mainNav").addClass("navbar-shrink");
    } else {
      $("#mainNav").removeClass("navbar-shrink");
    }
  };
  // Collapse now if page is not at top
  navbarCollapse();
  // Collapse the navbar when page is scrolled
  $(window).scroll(navbarCollapse);

  var layer_names = ['block1_conv1', 'block1_conv2','block2_conv1', 'block2_conv2',
  'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4',
  'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4',
  'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4'];
  
  loadFeatureMaps(cur_layer_name);
  loadFilters(cur_layer_name);
  $(".front1").css("background", "hsla(20, 100%, 50%, 0.7)");
  $(".back1").css("background", "hsla(20, 100%, 50%, 0.7)");
  $(".right1").css("background", "hsla(20, 100%, 50%, 0.7)");
  $(".left1").css("background", "hsla(20, 100%, 50%, 0.7)");

  layer_names.forEach((layer_name) => {
    $("#"+layer_name).click(function(e){
      clearHighlighted();
      hightlightVGG(layer_name, 2);
      cur_layer_name = layer_name;
      $(this).addClass('btn-highlighted');
      loadFeatureMaps(layer_name);
      loadFilters(layer_name);
    });
    $("#"+layer_name).mouseover(function(e){
      hightlightVGG(layer_name, 1);
    });
    $("#"+layer_name).mouseout(function(e){
      hightlightVGG(layer_name, 0);
    });
  })

  $("#refresh1").click(function(e){
    demo_id ++;
    if(demo_id > max_demo_id){
      demo_id = 1;
    }
    refresh1();
  });

  refresh1();

  var $scroll_divs1 = $('#filters_div, #feature_maps_div');
  var sync = function(e){
      console.log("sync")
      var $other = $scroll_divs1.not(this).off('scroll'), other = $other.get(0);
      var percentage = this.scrollLeft / (this.scrollWidth - this.offsetWidth);
      other.scrollLeft = percentage * (other.scrollWidth - other.offsetWidth);
      setTimeout( function(){ $other.on('scroll', sync ); },2);
  }
  $scroll_divs1.on( 'scroll', sync);

  var $scroll_divs2 = $('#origin_div, #occlusion_div, #guided_backprop_div');
  var sync2 = function(e){
      console.log("sync2")
      var $other = $scroll_divs2.not(this).off('scroll'), other = $other.get(0), other2 = $other.get(1);
      var percentage = this.scrollLeft / (this.scrollWidth - this.offsetWidth);
      other.scrollLeft = percentage * (other.scrollWidth - other.offsetWidth);
      other2.scrollLeft = percentage * (other2.scrollWidth - other2.offsetWidth);
      setTimeout( function(){ $other.on('scroll', sync2 ); },2);
  }
  $scroll_divs2.on( 'scroll', sync2);

  $("#light-box").hide();
  $(".img-clickable").click(function(e){
    $("#light-box-img").attr({ "src": $(this).attr('src') });
    $("#light-box").show();
  });

  $("#light-box").click(function(e){
    $("#light-box").hide();
  });

  $("#prev-content").click(function(e){
    change_style_transfer(-1, 0);
  });
  $("#next-content").click(function(e){
    change_style_transfer(1, 0);
  });
  $("#prev-style").click(function(e){
    change_style_transfer(0, -1);
  });
  $("#next-style").click(function(e){
    change_style_transfer(0, 1);
  });

})(jQuery)); // End of use strict
