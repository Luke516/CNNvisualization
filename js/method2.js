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

$('#send').click(function(){
    var outputDiv =  document.getElementById('output');
    outputDiv.innerHTML = "";
    document.getElementById("confirm").style.display = "none";
    document.getElementById("loading").style.display = "flex";
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
    data.inputs = {"input": pixels};

    console.log(JSON.stringify(data));
    console.log(JSON.stringify(pixels));

    const proxyurl = "https://cors-anywhere.herokuapp.com/";

    const httpHeaders = { 'Content-Type' : 'application/json', 'X-Requested-With': 'XMLHttpRequest'}
    const myHeaders = new Headers(httpHeaders)
    const url = proxyurl + "http://140.114.85.24:8501/v1/models/model:predict";
    const req = new Request(url, {method: 'POST', headers: myHeaders})

    fetch(url, {
        method: 'post',
        headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
        },
        // body: '{"signature_name":"activation"}'
        body: JSON.stringify(data)
    }).then(res=>res.json())
    .then((res) => {
        document.getElementById("confirm").style.display = "flex";
        document.getElementById("loading").style.display = "none";
        var root = res['outputs'];
        console.log(res);
        for(var key in root) {
            var outputRow = document.createElement('div');
            outputRow.setAttribute("class", "row");
            outputDiv.appendChild(outputRow);
            var outputTitle = key + " 的輸出 (前8個activation map)："
            var outputTitleElem = document.createElement('p');
            outputTitleElem.innerHTML = outputTitle;
            outputRow.appendChild(outputTitleElem);
            outputRow.appendChild(document.createElement('br'));
            if(root.hasOwnProperty(key)) {
                //var flat = flatten(root[key]);
                for(var i=0; i<8; i++){
                    var flat = getFilter(root[key][0], i);
                    console.log(flat);

                    var new_canvas = document.createElement('canvas');
                    new_canvas.id = key;
                    new_canvas.width = w;
                    new_canvas.height = h;
                    new_canvas.style.zIndex = 8;
                    new_canvas.style.position = "relative";
                    new_canvas.style.border = "1px solid";
                    //outputRow.appendChild(new_canvas);

                    var ctx = new_canvas.getContext("2d");
                    var resultImgData = ctx.createImageData(w, h);
                    resultImgData.data.set(Uint8ClampedArray.from(flat));
                    //QWQ
                    console.log(resultImgData.width);
                    console.log(resultImgData.height);
                    console.log(resultImgData.data);
                    ctx.putImageData(resultImgData,0,0);

                    var outputUrl = new_canvas.toDataURL();
                    imageFoo = document.createElement('img');
                    imageFoo.src = outputUrl;
                    imageFoo.style.maxWidth = '256px';
                    imageFoo.style.maxHeight = '256px';
                    outputRow.appendChild(imageFoo);
                }
                // break;
            }
        }
    });  

    // $.ajax({
    //     type: "POST",
    //     url: "https://cors-anywhere.herokuapp.com/http://140.114.85.24:8501/v1/models/model:predict",
    //     contentType: 'application/json',
    //     data: JSON.stringify(data),
    //     success: function (msg) {
    //         console.log(msg);
    //     },
    //     error: function(XMLHttpRequest, textStatus, errorThrown) {
    //         console.log(XMLHttpRequest.status);
    //         console.log(XMLHttpRequest.readyState);
    //         console.log(textStatus);
    //         console.log(errorThrown)
    //     }
    // });
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
