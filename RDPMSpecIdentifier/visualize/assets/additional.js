window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {

        restyleRadio: function(url) {
            var formCheckElements = document.getElementsByClassName("form-check");
            if (formCheckElements.length > 0) {
                for (var i = 0; i < formCheckElements.length; i++) {
                    formCheckElements[i].classList.add("col-4");
                    formCheckElements[i].classList.add("col-md-3");
                }
            }
            return ""


        },

        moveBtn: function (tbl, tab) {
            if (tab === "tab-2") {
                return ""
            } else {
                var btn = document.getElementById("reset-rows-btn")
                var container = document.getElementsByClassName("previous-next-container")[0];
                container.insertBefore(btn, container.firstChild);
                return ""
            }

        },

        styleSelectedTableRow: function (proteinKey, tbl_data) {
            const key = proteinKey.split("Protein ")[1];
            console.log("Searching for the Protein Key:" + key)
            try {
                var tables = document.getElementById("tbl").querySelectorAll("table");

            } catch (error) {
                return ""
            }
            var table = tables[[tables.length - 1]];
            var rows = table.getElementsByTagName("tr");
            for (let i = 0; i < rows.length; i++) {
                var cells = rows[i].getElementsByTagName("td");
                rows[i].classList.remove('selected-row');

                // Check if the first column value (cells[0]) matches the custom key
                //console.log(cells[1].children[0].textContent)
                if (cells.length > 0 && cells[1].children[0].textContent === key) {
                    // You've found a match, you can do something with it
                    console.log("Found a match at row " + (i + 1));
                    rows[i].classList.add("selected-row")
                }
            }
            return ""


        },

        styleFlamingo: function (on, style, fill_start, black_start) {
            const svgImage = document.getElementById('flamingo-svg');
            style = this.rgbToHex(style)
            const fill = "fill:" + style;
            console.log(fill)
            if (svgImage) {
                const base64EncodedSvg = svgImage.getAttribute('src').replace(/^data:image\/svg\+xml;base64,/, '');
                const decodedSvg = atob(base64EncodedSvg);
                var modifiedSvg = this.substrReplace(decodedSvg, fill_start, fill);
                // if (!on) {
                //     modifiedSvg = this.substrReplace(modifiedSvg, black_start, "fill:#000000");
                //
                // } else {
                //     modifiedSvg = this.substrReplace(modifiedSvg, black_start, "fill:#f2f2f2");
                //
                // }
                svgImage.setAttribute('src', 'data:image/svg+xml;base64,' + btoa(modifiedSvg));
            }
            return ""
        },

        rgbToHex: function rgbToHex(rgbString) {
            const match = rgbString.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/);

            if (!match) {
                throw new Error('Invalid RGB string format');
            }

            const [, red, green, blue] = match;

            const hexRed = parseInt(red).toString(16).padStart(2, '0');
            const hexGreen = parseInt(green).toString(16).padStart(2, '0');
            const hexBlue = parseInt(blue).toString(16).padStart(2, '0');

            return `#${hexRed}${hexGreen}${hexBlue}`;
        },

        substrReplace: function replaceInRange(inputString, startCoordinate, replacement) {
            const startIndex = startCoordinate;
            const endIndex = startIndex + replacement.length;

            const newString = inputString.slice(0, startIndex) + replacement + inputString.slice(endIndex);

            return newString;
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

        nightMode: function changeMode(on, primaryColor, secondaryColor) {
            console.log("Running night mode function")
            var r = document.querySelector(':root');
            console.log(primaryColor)
            r.style.setProperty('--primary-color', primaryColor)
            r.style.setProperty('--secondary-color', secondaryColor)

            if (on) {
                r.style.setProperty('--r-text-color', "white")
                r.style.setProperty('--databox-color', "#181818")
                r.style.setProperty('--table-light', "#3a363d")
                r.style.setProperty('--table-dark', "#222023")
                r.style.setProperty('--button-color', "#222023")
                r.style.setProperty('--input-background-color', "#2f2f2f")
                r.style.setProperty('--background-color', "#3a3a3a")
                r.style.setProperty('--disabled-input', "#181818")
                var darker = this.function2(primaryColor, 0.5);
                var darker2 = this.function2(secondaryColor, 0.5);
                r.style.setProperty('--primary-hover-color', darker);
                r.style.setProperty('--secondary-hover-color', darker2);
                var table_head = this.function2(primaryColor, 0.05);
                r.style.setProperty('--table-head-color', table_head);
                console.log("Setup Night Mode")


            } else {
                r.style.setProperty('--r-text-color', "black")
                r.style.setProperty('--databox-color', "#fffdfd")
                r.style.setProperty('--table-light', "#e1e1e1")
                r.style.setProperty('--table-dark', "#c0c0c0")
                r.style.setProperty('--button-color', "#8f8f8f")
                r.style.setProperty('--input-background-color', "#e0e0e0")
                r.style.setProperty('--background-color', "#ffffff")
                var lighter = this.makebrighter(primaryColor, 50);
                var lighter2 = this.makebrighter(secondaryColor, 50);
                r.style.setProperty('--table-head-color', "#181818");
                r.style.setProperty('--primary-hover-color', lighter);
                r.style.setProperty('--secondary-hover-color', lighter2);
                r.style.setProperty('--disabled-input', "#a6a6a6")

                console.log("Setup Day Mode")




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
//
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
//
//
//
// document.addEventListener('focus', function (event) {
//     const focusedElement = event.target;
//     if (focusedElement.tagName === 'TD') {
//         const elements = document.getElementsByClassName('selected-row'); // Replace with your class name
//         for (let i = 0; i < elements.length; i++) {
//             console.log(elements[i])
//             elements[i].classList.remove('selected-row');
//         }
//         const row = focusedElement.parentNode
//         row.classList.add("selected-row")
//
//         // Add your code to run when a table cell gains focus
//     }
//     // Add your code to run when an element gains focus
// }, true);


addEventListener("dragover", (event) => {
    const dragOverElement = event.target;
    if (dragOverElement.classList.contains("custom-tab")) {
        dragOverElement.click()
    }
});

var btn = document.getElementById("reset-rows-btn")
var container = document.getElementsByClassName("previous-next-container")[0];
container.insertBefore(btn, container.firstChild);