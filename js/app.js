function loadCSS(file){
    if($('link[href="' + file + '"]').size() > 0){ 
        console.log('already loaded ' + file);
    }else{
      if (document.createStyleSheet){
        document.createStyleSheet( file );
      }
      else {
        $("head").append($('<link rel="stylesheet" href="' + file +'" type="text/css" />'));
      }
    }
}
