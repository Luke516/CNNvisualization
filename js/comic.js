var cur_page = 0;
var max_page = 0;
var comic_pages = [];

$( document ).ready((function($) {
    
//   $('.carousel-item').each(
//          function(index,ele) { 
//             console.log($(ele).find('.comic-cell').length);
//             if($(ele).find('.comic-cell').length > 0){
//                 comic_pages.push(index);
//                 console.log(comic_pages);
//             }
//             // $(ele).attr("debog",index).data("pos",index);
//             max_page = max_page + 1;
//         }
//     );

    $('.comic-section').find('.comic-cell').addClass('invisible');
    $('.comic-section').find('>:first-child').prepend(
        '<div class="col-12 image-block flex-grow my-2 animated fadeIn click-me"><p>點擊滑鼠來觀看對話！</p>' +
        '<a href="#" class="show-all">顯示全部</a></div>'
    );

    // $('.comic-section').find('.comic-cell').first().removeClass('invisible');
    // $('.comic-section').find('.comic-cell').first().addCalss('animated fadeIn');

    $('.show-all').on('click', function(e){
        console.log('click QxQ');
        $(this).parent().parent().find('.invisible').removeClass('invisible');
        $(this).remove();
    })

    $('.comic-section').on('click', function(e){
        // console.log('click QWQ');
        // $(self).find('.comic-cell invisible').first().addCalss('animated fadeIn');
        // $('.comic-section').find('.invisible').first().removeClass('invisible');
        $(this).find('.click-me').remove();
        $(this).find('.invisible').first().addClass('animated fadeIn');
        $(this).find('.invisible').first().removeClass('invisible');
    })

    $('.carousel-control-prev').click(function(e){
        cur_page = cur_page - 1;
        if(cur_page < 0){
            cur_page = 0;
        }
        console.log(cur_page);
    })

    $('.carousel-control-next').click(function(e){
        cur_page = cur_page + 1;
        if(cur_page == max_page){
            cur_page = max_page - 1;
        }
        console.log(cur_page);
    })

})(jQuery)); // End of use strict
