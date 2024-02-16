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
                if (cells.length > 0 && cells[1].children[0].textContent === key) {
                    // You've found a match, you can do something with it
                    rows[i].classList.add("selected-row")
                }
            }
            return ""


        },
        styleTutorial: function (style2, style, fill_starts, fill2_starts) {
            const svgImage = document.getElementById('tutorial-rapdor-svg');
            style = this.rgbToHex(style)
            style2 = this.rgbToHex(style2)
            const fill = "fill:" + style;
            const fill2 = "fill:" + style2;
            if (svgImage) {

                const base64EncodedSvg = svgImage.getAttribute('src').replace(/^data:image\/svg\+xml;base64,/, '');
                const decodedSvg = atob(base64EncodedSvg);
                var modifiedSvg = decodedSvg;

                // Iterate over each index in fill_starts and apply substrReplace
                fill_starts.forEach(fill_start => {
                    modifiedSvg = this.substrReplace(modifiedSvg, fill_start, fill);
                });
                fill2_starts.forEach(fill_start => {
                    modifiedSvg = this.substrReplace(modifiedSvg, fill_start, fill2);
                });
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

        styleFlamingo: function (style2, style, fill_starts, fill2_starts) {
            const svgImage = document.getElementById('flamingo-svg');
            style = this.rgbToHex(style)
            style2 = this.rgbToHex(style2)
            const fill = "fill:" + style;
            const fill2 = "fill:" + style2;
            if (svgImage) {

                const base64EncodedSvg = svgImage.getAttribute('src').replace(/^data:image\/svg\+xml;base64,/, '');
                const decodedSvg = atob(base64EncodedSvg);
                var modifiedSvg = decodedSvg;

                // Iterate over each index in fill_starts and apply substrReplace
                fill_starts.forEach(fill_start => {
                    modifiedSvg = this.substrReplace(modifiedSvg, fill_start, fill);
                });
                fill2_starts.forEach(fill_start => {
                    modifiedSvg = this.substrReplace(modifiedSvg, fill_start, fill2);
                });
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

        displayToolTip: function displayEllipsies(input_trigger) {
            var elements = document.querySelectorAll('.column-header-name');
            console.log(elements)
            elements.forEach(function (element) {
                    element.addEventListener('mouseover', function (event) {
                        if (element.scrollWidth > element.clientWidth) {
                            console.log(element.scrollWidth)
                            console.log(element.clientWidth)

                            var fullText = element.textContent;
                            var tooltip = document.createElement('div');
                            tooltip.textContent = fullText;
                            tooltip.classList.add('rtooltip');
                            tooltip.classList.add('databox');
                            document.body.appendChild(tooltip);

                            var x = event.pageX + 10; // Add 10px offset to avoid covering the mouse pointer
                            var y = event.pageY + 10;
                            tooltip.style.top = y + 'px';
                            tooltip.style.left = x + 'px';
                            tooltip.style.display = "block";
                        }
                    });

                    element.addEventListener('mouseout', function (event) {
                        var tooltip = document.querySelector('.rtooltip');
                        if (tooltip) {
                            tooltip.remove();
                        }
                    });
                })
            var elements = document.querySelectorAll('.dash-cell');
            elements.forEach(function (element) {
                    element.addEventListener('mouseover', function (event) {
                        if (element.scrollWidth > element.clientWidth) {
                            console.log(element.scrollWidth)
                            console.log(element.clientWidth)

                            var fullText = element.childNodes[0].textContent;
                            var tooltip = document.createElement('div');
                            tooltip.textContent = fullText;
                            tooltip.classList.add('rtooltip');
                            tooltip.classList.add('databox');
                            document.body.appendChild(tooltip);

                            var x = event.pageX + 10; // Add 10px offset to avoid covering the mouse pointer
                            var y = event.pageY + 10;
                            tooltip.style.top = y + 'px';
                            tooltip.style.left = x + 'px';
                            tooltip.style.display = "block";
                        }
                    });

                    element.addEventListener('mouseout', function (event) {
                        var tooltip = document.querySelector('.rtooltip');
                        if (tooltip) {
                            tooltip.remove();
                        }
                    });
                })

            return ""
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
            var r = document.querySelector(':root');
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





            }
            return ""
        },
        activateTutorial: function (btn, skip_btn) {
            const tutRow = document.getElementById('tut-overlay');
            const overlay = document.getElementById('tut-row');
            tutRow.classList.toggle('d-none');
            overlay.classList.toggle('d-none');
            return ""

        },
        tutorialStep: function (next, previous) {
            var ts = sessionStorage.getItem("tutorial-step");
            if (ts === null || ts === undefined) {
                // Set ts to zero
                ts = 0;
            }
            var highlightedElements = document.querySelectorAll('.highlighted');
            console.log(highlightedElements)
            highlightedElements.forEach(function (element) {
                element.classList.remove('highlighted');
            });
            ts = parseInt(ts);
            console.log(ts)
            if (this.tutorialSteps.hasOwnProperty(ts)) {
                // Key ts exists in this.tutorialSteps
                var [highlightID, page, text] = this.tutorialSteps[ts];
                ts = ts + 1
                sessionStorage.setItem("tutorial-step", ts);


                if (highlightID) {
                    console.log("highlighting")
                    var highlight = document.getElementById(highlightID);
                    console.log(highlight)
                    if (highlight) {
                        highlight.classList.toggle('highlighted');
                    }

                }
                if (page) {
                    //window.location.href = page


                }
            } else {
                // Key ts does not exist in this.tutorialSteps
                sessionStorage.setItem("tutorial-step", 0);
            }


            return 5

        },
        tutorialSteps: {
            0: [null, ".", null],
            1: ["intensities-row", null, null]
        }

    }

});



document.addEventListener('keydown', (event) => {
    const currentInput = document.getElementsByClassName("dash-cell focused")[0];
    console.log(currentInput)
    if (!currentInput) {
    // Break the code execution
    return ""
    // You may want to add any further actions here
    }
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
        const parent = clickedElement.parentNode;
        parent.focus();
        // Add your code to handle the click on the <div> inside the <td>
    } else if (clickedElement.tagName === 'INPUT' && clickedElement.closest('td')) {
        const parent = clickedElement.parentNode.nextElementSibling;
        document.activeElement.blur()
    }
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


