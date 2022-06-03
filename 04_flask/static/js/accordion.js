window.addEventListener('DOMContentLoaded', function(){
  document.querySelectorAll('details').forEach(function(item){
      item.addEventListener("toggle", event => {
      let toggled = event.target;
      if (toggled.attributes.open) {/* 열었으면 */
        /* 나머지 다른 열린 아이템을 닫음 */
        document.querySelectorAll('details[open]').forEach(function(opened){
            if(toggled != opened) /* 현재 열려있는 요소가 아니면 */
              opened.removeAttribute('open'); /* 열림 속성 삭제 */
        });
      }
    })
  });
});