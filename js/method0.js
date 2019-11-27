var scale, w, h, l;

function scaleToFit(canvas, img){
    // get the scale
    var scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    // get the top left position of the image
    var x = (canvas.width / 2) - (img.width / 2) * scale;
    var y = (canvas.height / 2) - (img.height / 2) * scale;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, parseInt(img.width * scale), parseInt(img.height * scale));
    return scale;
}

function flatten(arr) {
    return arr.reduce(function (flat, toFlatten) {
      return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
    }, []);
}

function getFilter(root, filterId) {
    var out = [];
    for(var i=0; i<h; i++){
        for(var j=0; j<w; j++){
            // console.log(i+','+j);
            out.push(root[i][j][filterId]);
            out.push(root[i][j][filterId]);
            out.push(root[i][j][filterId]);
            // out.push(100);
            // out.push(100);
            // out.push(100);
            out.push(255);
        }
    }
    return out;
}

// filter_count = {"block1":64, "block2":128, "block3":256, "block4":512, "block5":512};
filter_count = {"block1":64, "block2":64, "block3":64, "block4":64, "block5":64};

$('#send').click(function(){
    var block_id = $('#block-id').val();
    var conv_id = $('#conv-id').val();
    // $('#send').hide();
    if(block_id=="block1" || block_id=="block2"){
        if(conv_id=="conv3" || conv_id=="conv4"){
            alert(block_id + "_" + conv_id + " does not exist!");
            return;
        }
    }

    console.log(block_id);
    console.log(conv_id);
    console.log(filter_count[block_id]);
    $("#filter-visualization-block").empty();
    for(var i=0; i<filter_count[block_id]; i++){
        var filter_img = document.createElement("img");
        filter_img.setAttribute("class", "mx-1 my-1");
        filter_img.setAttribute("src", "../images/demo_imgs/filter/" + block_id + "_" + conv_id + "_" + i + ".png");
        filter_img.setAttribute("width", "128px");
        $("#filter-visualization-block").append(filter_img);
    }
})

var main = function() {
  $('#uploaded_img').change(function() {
    var img = new Image();
    var canvas = document.querySelector("canvas");
    var ctx = canvas.getContext("2d");
    img.onload = function(){
        //ctx.drawImage(img,0,0);
        scale = scaleToFit(canvas, img);
        w = parseInt(img.width * scale);
        h = parseInt(img.height * scale);
        l = w * h;
    }
    img.src = window.URL.createObjectURL(this.files[0]);
  });
};

$(document).ready(main);
