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

$('#send').click(function(){
    var canvas = document.querySelector("canvas");
    var ctx = canvas.getContext("2d");
    var pixels = [];

    var imgData = ctx.getImageData(0, 0, w, h).data;
    //ctx.putImageData(imgData, 10, 70);
    //console.log(JSON.stringify(imgData));
    console.log(w);
    console.log(h);
    
    pixels.push([]);
    for(var i=0; i<h; i++){
        pixels[0].push([]);
    }
    for (var i = 0; i < l; i++) {
        // get color of pixel
        // get the position of pixel
        var y = parseInt(i / w, 10);
        var x = i - y * w;

        pixels[0][y].push([
            // imgData[i*4],
            // imgData[i*4+1],
            // imgData[i*4+2]
            Number.parseFloat((imgData[i*4+2] - 103.939).toFixed(5)),// Red
            Number.parseFloat((imgData[i*4+1] - 116.779).toFixed(5)), // Green
            Number.parseFloat((imgData[i*4] - 123.68).toFixed(5)) // Blue
        ]);
    }
    var data = Object();
    data.signature_name = "activation";
    data.instance = pixels;

    console.log(JSON.stringify(data));
    // console.log(JSON.stringify(pixels));

    const proxyurl = "https://cors-anywhere.herokuapp.com/";
    $.ajax({
        type: "POST",
        dataType: "json",
        url: proxyurl + "http://140.114.85.24:8501/v1/models/model:predict",
        contentType: 'application/json',
        data: data,
        success: function (msg) {
            console.log(msg);
        }
    });
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
