var curSlide = 1
var maxSlide = 6

function nextSlide(){

}


function prevSlide(){
    
}

$('.jump-page').keypress(function(e){
    code = (e.keyCode ? e.keyCode : e.which);
    if (code == 13)
    {
        console.log(e.target.value);
        curSlide = parseInt(e.target.value);
        if(Number.isNaN(curSlide) ){
            curSlide = 0;
        }
        curSlide = Math.max(curSlide, 1);
        curSlide = Math.min(curSlide, maxSlide);
        $('#carouselExampleControls').carousel(curSlide - 1);
        $('#jump-page').val(String(curSlide));
    }
});

$(".carousel-control-prev").click(function(e){
    if($("#toggle-comic").prop("checked")){
        var tmpSlide = curSlide - 1;
        var curSlideElement = $('.carousel-item').get(tmpSlide-1);
        var isComic = $(curSlideElement).find(".comic-section").length > 0;
        while($("#toggle-comic").prop("checked") && isComic){
            tmpSlide = tmpSlide - 1;
            if(tmpSlide < 1){
                break;
            }
            curSlideElement = $('.carousel-item').get(curSlide-1);
            isComic = $(curSlideElement).find(".comic-section").length > 0;
        }
        if(tmpSlide > 0){
            curSlide = tmpSlide;
            $('#carouselExampleControls').carousel(curSlide - 1);
            $('#jump-page').val(String(curSlide));
        }
        return;
    }
    curSlide = curSlide - 1;
    curSlide = Math.max(curSlide, 1);
    $('#carouselExampleControls').carousel(curSlide - 1);
    $('#jump-page').val(String(curSlide));
});

$(".carousel-control-next").click(function(e){
    if($("#toggle-comic").prop("checked")){
        var tmpSlide = curSlide + 1;
        var curSlideElement = $('.carousel-item').get(tmpSlide-1);
        var isComic = $(curSlideElement).find(".comic-section").length > 0;
        while($("#toggle-comic").prop("checked") && isComic){
            tmpSlide = tmpSlide + 1;
            if(tmpSlide > maxSlide){
                break;
            }
            curSlideElement = $('.carousel-item').get(curSlide-1);
            isComic = $(curSlideElement).find(".comic-section").length > 0;
        }
        if(tmpSlide <= maxSlide){
            curSlide = tmpSlide;
            $('#carouselExampleControls').carousel(curSlide - 1);
            $('#jump-page').val(String(curSlide));
        }
        return;
    }
    curSlide = curSlide + 1;
    curSlide = Math.min(curSlide, maxSlide);
    $('#carouselExampleControls').carousel(curSlide - 1);
    $('#jump-page').val(String(curSlide));
});

$("#toggle-comic").click(function(e){
    var curSlideElement = $('.carousel-item').get(curSlide-1);
    var isComic = $(curSlideElement).find(".comic-section").length > 0;
    while(e.target.checked && isComic){
        curSlide = curSlide + 1;
        if(curSlide > maxSlide){
            curSlide = maxSlide;
            break;
        }
        curSlideElement = $('.carousel-item').get(curSlide-1);
        isComic = $(curSlideElement).find(".comic-section").length > 0;
    }

    while(e.target.checked && isComic){
        curSlide = curSlide - 1;
        if(curSlide < 1){
            curSlide = 1;
            break;
        }
        curSlideElement = $('.carousel-item').get(curSlide-1);
        isComic = $(curSlideElement).find(".comic-section").length > 0;
    }

    $('#carouselExampleControls').carousel(curSlide - 1);
    $('#jump-page').val(String(curSlide));
});

function findGetParameter(parameterName) {
    var result = null,
        tmp = [];
    var items = location.search.substr(1).split("&");
    for (var index = 0; index < items.length; index++) {
        tmp = items[index].split("=");
        if (tmp[0] === parameterName) result = decodeURIComponent(tmp[1]);
    }
    return result;
}

$(document).ready(() => {
    curSlide = 1;
    $('#jump-page').val(String(curSlide));
    maxSlide = $('.carousel-item').length;
    maxSlide = Math.max(maxSlide, 1);
    $('#total-page').html("第"+ String(maxSlide) +"頁");

    var dst = parseInt(findGetParameter("page"));
    if(dst > 0 && dst <= maxSlide){
        curSlide = dst
        $('#carouselExampleControls').carousel(curSlide - 1);
        $('#jump-page').val(String(curSlide));
    }
});

