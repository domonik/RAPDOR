window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {

        function1: function (on, style, style2) {
            // Get the root element

            var r = document.querySelector(':root');
            r.style.setProperty('--primary-color', style["background-color"])
            r.style.setProperty('--secondary-color', style2["background-color"])

            if (on) {
                var darker = this.function2(style["background-color"], 0.5);
                r.style.setProperty('--primary-hover-color', darker);
                var table_head = this.function2(style["background-color"], 0.05);
                r.style.setProperty('--table-head-color', table_head);
            } else {
                var lighter = this.makebrighter(style["background-color"], 50);
                r.style.setProperty('--table-head-color', "#181818");
                r.style.setProperty('--primary-hover-color', lighter);

            }


            return "";
        },

        function2: function modifyRGB(inputRGB, multiplier) {
            const valuesStr = inputRGB.substring(inputRGB.indexOf("(") + 1, inputRGB.indexOf(")")).split(",");
            const values = [];
            for (let i = 0; i < valuesStr.length; i++) {
                values[i] = parseInt(valuesStr[i].trim());
                values[i] = Math.round(values[i] * multiplier);
            }

            return `rgb(${values[0]}, ${values[1]}, ${values[2]})`;
        },
        makebrighter: function makeRGBBrighter(inputRGB, percentage) {
            const valuesStr = inputRGB.substring(inputRGB.indexOf("(") + 1, inputRGB.indexOf(")")).split(",");
            const values = [];

            for (let i = 0; i < valuesStr.length; i++) {
                values[i] = parseInt(valuesStr[i].trim());
            }

            const diffR = 255 - values[0];
            const diffG = 255 - values[1];
            const diffB = 255 - values[2];

            const brighterR = Math.round(diffR * (percentage / 100));
            const brighterG = Math.round(diffG * (percentage / 100));
            const brighterB = Math.round(diffB * (percentage / 100));

            const newR = values[0] + brighterR;
            const newG = values[1] + brighterG;
            const newB = values[2] + brighterB;

            return `rgb(${newR}, ${newG}, ${newB})`;
        },

        nightMode: function changeMode(on) {
            var r = document.querySelector(':root');

            if (on) {
                r.style.setProperty('--r-text-color', "white")
                r.style.setProperty('--databox-color', "#181818")
                r.style.setProperty('--table-light', "#3a363d")
                r.style.setProperty('--table-dark', "#222023")
                r.style.setProperty('--button-color', "#222023")
                r.style.setProperty('--input-background-color', "#1a1a1a")
                r.style.setProperty('--input-background-color', "#1a1a1a")
                r.style.setProperty('--background-color', "#3a3a3a")


            } else {
                r.style.setProperty('--r-text-color', "black")
                r.style.setProperty('--databox-color', "#fffdfd")
                r.style.setProperty('--table-light', "#e1e1e1")
                r.style.setProperty('--table-dark', "#c0c0c0")
                r.style.setProperty('--button-color', "#8f8f8f")
                r.style.setProperty('--input-background-color', "#9a9a9a")
                r.style.setProperty('--background-color', "#ffffff")



            }
            return ""
        }

    }

});

document.addEventListener('keydown', (event) => {
    const currentInput = document.getElementsByClassName("dash-cell focused")[0];
    const currentTr = currentInput.parentNode;
    switch (event.key) {
        case "ArrowUp":
            // Up pressed

            (currentTr.previousElementSibling.children[1]).focus();
            break;
        case "ArrowDown":
            // Down pressed
            (currentTr.nextElementSibling.children[1]).focus();
            break;
    }
})

document.addEventListener('click', (event) => {
    const clickedElement = event.target;

    // Check if the clicked element is a <div> inside a <td>
    if (clickedElement.tagName === 'DIV' && clickedElement.closest('td')) {
        console.log('Clicked element is a <div> inside a <td>');
        const parent = clickedElement.parentNode;
        parent.focus();
        // Add your code to handle the click on the <div> inside the <td>
    } else if (clickedElement.tagName === 'INPUT' && clickedElement.closest('td')) {
        const parent = clickedElement.parentNode.nextElementSibling;
        console.log("parent", parent)
        document.activeElement.blur()
    }
    console.log(clickedElement);
})


async function styleCell() {
    await new Promise(resolve => {
        setTimeout(() => {
            // Add the code you want to execute after the delay
            resolve(); // Resolve the promise after the delay
        }, 300);
    });
    console.log("lallala")
    const currentInput = document.getElementsByClassName("dash-cell focused")[0];
    const cells = currentInput.parentNode.children;
    for (let i = 0; i < cells.length; i++) {
        const cell = cells[i];
        cell.style.border = '2px solid black'; // Add border styling
    }
}

document.addEventListener('focus', function (event) {
    const focusedElement = event.target;
    if (focusedElement.tagName === 'TD') {
        const elements = document.getElementsByClassName('selected-row'); // Replace with your class name
        for (let i = 0; i < elements.length; i++) {
            console.log(elements[i])
            elements[i].classList.remove('selected-row');
        }
        const row = focusedElement.parentNode
        row.classList.add("selected-row")

        // Add your code to run when a table cell gains focus
    }
    // Add your code to run when an element gains focus
}, true);
