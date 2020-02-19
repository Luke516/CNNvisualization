/* jshint esnext: true */
/* global document */
let canvas = document.getElementById("pixel_canvas");
let canvas2 = document.getElementById("pixel_canvas-red");
let canvas3 = document.getElementById("pixel_canvas-green");
let canvas4 = document.getElementById("pixel_canvas-blue");
let canvas5 = document.getElementById("pixel_canvas-red-filter");
let canvas6 = document.getElementById("pixel_canvas-green-filter");
let canvas7 = document.getElementById("pixel_canvas-blue-filter");
let canvas8 = document.getElementById("pixel_canvas-red-output");
let canvas9 = document.getElementById("pixel_canvas-green-output");
let canvas10 = document.getElementById("pixel_canvas-blue-output");
let canvas11 = document.getElementById("pixel_canvas-final-output");
// let canvas5 = document.getElementById("pixel_canvas-red-value");
// let canvas6 = document.getElementById("pixel_canvas-green-value");
// let canvas7 = document.getElementById("pixel_canvas-blue-value");
let height = document.getElementById("input_height");
let width = document.getElementById("input_width");
let sizePicker = document.getElementById("sizePicker");
let color = document.getElementById("colorPicker");

let filter_width = 3;
let filter_height = 3;

let filter = [[0,255,0],[0,255,0],[0,255,0]];

color.addEventListener("click", function(){});

sizePicker.onsubmit = function(event){
    event.preventDefault();
    clearGrid();
    makeGrid();
};

function makeGrid() {
    for (let r=0; r<height.value; r++){
        const row = canvas.insertRow(r);
        for (let c=0; c<width.value; c++){
            const cell = row.insertCell(c);
            cell.setAttribute("style", `background-color: #000000`);
            // cell.addEventListener("click", fillSquare);
            $(cell).click(fillSquare);
        }
    }

    for (let r=0; r<height.value; r++){
        const row = canvas2.insertRow(r);
        for (let c=0; c<width.value; c++){
            const cell = row.insertCell(c);
            cell.setAttribute("style", `background-color: #000000`);
            $(cell).html(0);
            $(cell).attr("id", r+"-"+c+"r");
        }
    }

    for (let r=0; r<height.value; r++){
        const row = canvas3.insertRow(r);
        for (let c=0; c<width.value; c++){
            const cell = row.insertCell(c);
            cell.setAttribute("style", `background-color: #000000`);
            $(cell).html(0);
            $(cell).attr("id", r+"-"+c+"g");
        }
    }

    for (let r=0; r<height.value; r++){
        const row = canvas4.insertRow(r);
        for (let c=0; c<width.value; c++){
            const cell = row.insertCell(c);
            cell.setAttribute("style", `background-color: #000000`);
            $(cell).html(0);
            $(cell).attr("id", r+"-"+c+"b");
        }
    }

    for (let r=0; r<filter_height; r++){
        const row = canvas5.insertRow(r);
        for (let c=0; c<filter_width; c++){
            const cell = row.insertCell(c);
            $(cell).html((filter[r][c]/255/3).toFixed(1));
            $(cell).css("background-color", "#"+filter[r][c].toString(16)+"0000");
            $(cell).css("text-align", "center");
            $(cell).attr("id", r+"-"+c+"rf");
            $("#multiplier1-"+String(r*3+c+1)).html(" * "+(filter[r][c]/255/3).toFixed(1)+" = ");
        }
    }

    for (let r=0; r<filter_height; r++){
        const row = canvas6.insertRow(r);
        for (let c=0; c<filter_width; c++){
            const cell = row.insertCell(c);
            $(cell).html((filter[r][c]/255/3).toFixed(1));
            $(cell).css("background-color", "#00"+filter[r][c].toString(16)+"00");
            $(cell).css("text-align", "center");
            $(cell).attr("id", r+"-"+c+"gf");
            $("#multiplier2-"+String(r*3+c+1)).html(" * "+(filter[r][c]/255/3).toFixed(1)+" = ");
        }
    }

    for (let r=0; r<filter_height; r++){
        const row = canvas7.insertRow(r);
        for (let c=0; c<filter_width; c++){
            const cell = row.insertCell(c);
            $(cell).html((filter[r][c]/255/3).toFixed(1));
            $(cell).css("background-color", "#0000"+filter[r][c].toString(16));
            $(cell).css("text-align", "center");
            $(cell).attr("id", r+"-"+c+"bf");
            $("#multiplier3-"+String(r*3+c+1)).html(" * "+(filter[r][c]/255/3).toFixed(1)+" = ");
        }
    }

    for (let r=0; r<height.value-2; r++){
        const row = canvas8.insertRow(r);
        for (let c=0; c<width.value-2; c++){
            const cell = row.insertCell(c);
            cell.setAttribute("style", `background-color: #000000`);
            $(cell).html(0);
            $(cell).attr("id", r+"-"+c+"ro");
        }
    }

    for (let r=0; r<height.value-2; r++){
        const row = canvas9.insertRow(r);
        for (let c=0; c<width.value-2; c++){
            const cell = row.insertCell(c);
            cell.setAttribute("style", `background-color: #000000`);
            $(cell).html(0);
            $(cell).attr("id", r+"-"+c+"go");
        }
    }

    for (let r=0; r<height.value-2; r++){
        const row = canvas10.insertRow(r);
        for (let c=0; c<width.value-2; c++){
            const cell = row.insertCell(c);
            cell.setAttribute("style", `background-color: #000000`);
            $(cell).html(0);
            $(cell).attr("id", r+"-"+c+"bo");
        }
    }

    for (let r=0; r<height.value-2; r++){
        const row = canvas11.insertRow(r);
        for (let c=0; c<width.value-2; c++){
            const cell = row.insertCell(c);
            cell.setAttribute("style", `background-color: #000000`);
            $(cell).html(0);
            $(cell).attr("id", r+"-"+c+"fo");
        }
    }
}

function clearGrid(){
    while (canvas.firstChild){
        canvas.removeChild(canvas.firstChild);
        canvas2.removeChild(canvas2.firstChild);
        canvas3.removeChild(canvas3.firstChild);
        canvas4.removeChild(canvas4.firstChild);
        // canvas5.removeChild(canvas5.firstChild);
        // canvas6.removeChild(canvas6.firstChild);
        // canvas7.removeChild(canvas7.firstChild);
    }
}
// alternative code:
// while (table.rows.length > 0) {
//  table.deleteRow(0);
// }

function getRGB(str){
var match = str.match(/rgba?\((\d{1,3}), ?(\d{1,3}), ?(\d{1,3})\)?(?:, ?(\d(?:\.\d?))\))?/);
    return match ? {
        red: match[1],
        green: match[2],
        blue: match[3]
    } : {};
}

function fillSquare () {
    // alert('Row ' + $(this).closest("tr").index());
    // alert('Column ' + $(this).closest("td").index());
    var selected_row = $(this).closest("tr").index();
    var selected_col = $(this).closest("td").index();
    this.setAttribute("style", `background-color: ${color.value}`);

    var r = String(color.value).substring(1,3);
    var g = String(color.value).substring(3,5);
    var b = String(color.value).substring(5,7);
    var z = "00";
    console.log(r+" "+g+" "+b);
    $("#"+selected_row+"-"+selected_col+"r").css("background-color", "#"+r+z+z);
    $("#"+selected_row+"-"+selected_col+"g").css("background-color", "#"+z+g+z);
    $("#"+selected_row+"-"+selected_col+"b").css("background-color", "#"+z+z+b);

    $("#"+selected_row+"-"+selected_col+"r").html(parseInt("0x"+r));
    $("#"+selected_row+"-"+selected_col+"g").html(parseInt("0x"+g));
    $("#"+selected_row+"-"+selected_col+"b").html(parseInt("0x"+b));
}

var conv_animation_row = 0;
var conv_animation_col = 0;

function clearHighlight(){
    for(let i=conv_animation_row; i<conv_animation_row+3; i++){
        for(let j=conv_animation_col; j<conv_animation_col+3; j++){
            $("#"+i+"-"+j+"r").css("border", "solid 1px white");
            $("#"+i+"-"+j+"g").css("border", "solid 1px white");
            $("#"+i+"-"+j+"b").css("border", "solid 1px white");
        }
    }
}

function highlightTable(){

    if(conv_animation_col > 7){
        conv_animation_col = 0;
        conv_animation_row = conv_animation_row + 1;
    }

    if(conv_animation_row > 7){
        conv_animation_col = 0;
        conv_animation_row = 0;
        return;
    }
    let sum_1 = 0;
    let sum_2 = 0;
    let sum_3 = 0;
    for(let i=0; i<3; i++){
        for(let j=0; j<3; j++){
            let id_r = "#"+(i+conv_animation_row)+"-"+(j+conv_animation_col)+"r";
            let id_g = "#"+(i+conv_animation_row)+"-"+(j+conv_animation_col)+"g";
            let id_b = "#"+(i+conv_animation_row)+"-"+(j+conv_animation_col)+"b";
            $(id_r).css("border", "solid 2px red");
            $(id_g).css("border", "solid 2px red");
            $(id_b).css("border", "solid 2px red");
            $("#multiple1-"+((i)*3+(j)+1)).html($(id_r).html());
            $("#multiple2-"+((i)*3+(j)+1)).html($(id_g).html());
            $("#multiple3-"+((i)*3+(j)+1)).html($(id_b).html());

            let result_1 = parseInt(parseInt($(id_r).html()) * filter[i][j]/255/3);
            let result_2 = parseInt(parseInt($(id_g).html()) * filter[i][j]/255/3);
            let result_3 = parseInt(parseInt($(id_b).html()) * filter[i][j]/255/3);
            $("#result1-"+((i)*3+(j)+1)).html(result_1);
            $("#result2-"+((i)*3+(j)+1)).html(result_2);
            $("#result3-"+((i)*3+(j)+1)).html(result_3);

            sum_1 = sum_1 + result_1;
            sum_2 = sum_2 + result_2;
            sum_3 = sum_3 + result_3;
        }
        $("#sum1").html(sum_1);
        $("#sum2").html(sum_2);
        $("#sum3").html(sum_3);
        $("#"+(conv_animation_row)+"-"+(conv_animation_col)+"ro").html(sum_1);
        $("#"+(conv_animation_row)+"-"+(conv_animation_col)+"go").html(sum_2);
        $("#"+(conv_animation_row)+"-"+(conv_animation_col)+"bo").html(sum_3);
        $("#"+(conv_animation_row)+"-"+(conv_animation_col)+"fo").html(parseInt((sum_1 + sum_2 + sum_3) / 3));
    }

    setTimeout(()=>{
        clearHighlight();
        conv_animation_col = conv_animation_col + 1;
        highlightTable();
    }, 1200);
}

$("#start-conv-demo").click(function(e){
    $("#start-conv-demo").hide();
    $(".demo-hide").show(); 
    highlightTable();
});

// document.onload(makeGrid());
$(document).ready(()=>{
    makeGrid();
    $(".demo-hide").hide(); 
});