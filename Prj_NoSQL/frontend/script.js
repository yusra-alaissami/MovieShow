document.addEventListener("DOMContentLoaded", async () => {
    const movieGrid = document.getElementById("movie-grid");

    // Fetch recommended movies
    const response = await fetch("http://127.0.0.1:5001/recommend?Inception");
    const movies = await response.json();

    // Display movies
    movies.forEach(movie => {
        const movieCard = document.createElement("div");
        movieCard.className = "movie";
        movieCard.innerHTML = `
            <img src="${movie.image}" alt="${movie.title}">
            <h3>${movie.title}</h3>
        `;
        movieGrid.appendChild(movieCard);
    });
});
