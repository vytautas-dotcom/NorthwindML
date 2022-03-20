using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using NorthwindML.Models;
using NorthwindDb;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Hosting;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data;
using Microsoft.ML.Trainers;

namespace NorthwindML.Controllers
{
    public class HomeController : Controller
    {
        private readonly static string datasetName = "dataset.txt";
        private readonly static string[] countries = new[] {"Germany", "UK", "USA"};
        private readonly ILogger<HomeController> _logger;
        private readonly Northwind _db;
        private readonly IWebHostEnvironment _webHostEnvironment;
        

        public HomeController(ILogger<HomeController> logger, 
                              Northwind db, 
                              IWebHostEnvironment webHostEnvironment)
        {
            _logger = logger;
            _db = db;
            _webHostEnvironment = webHostEnvironment;
        }

        public IActionResult Index()
        {
            var model = CreateHomeIndexViewModel();
            return View(model);
        }

        public IActionResult GenerateDatasets()
        {
            foreach(string country in countries)
            {
                IEnumerable<Order> ordersInCountry = _db.Orders
                    .Where(order => order.Customer.Country == country)
                    .Include(order => order.OrderDetails)
                    .AsEnumerable();

                IEnumerable<ProductCobought> coboughtProducts = ordersInCountry
                    .SelectMany(order => 
                    from lineItem1 in order.OrderDetails
                    from lineItem2 in order.OrderDetails
                    select new ProductCobought
                    {
                        ProductID = (uint)lineItem1.ProductID,
                        CoboughtProductID = (uint)lineItem2.ProductID
                    })
                    .Where(p => p.ProductID != p.CoboughtProductID)
                    .GroupBy(p => new {p.ProductID, p.CoboughtProductID})
                    .Select(p => p.FirstOrDefault())
                    .OrderBy(p => p.ProductID)
                    .ThenBy(p => p.CoboughtProductID);

                StreamWriter sw = System.IO.File.CreateText(path: GetDataPath($"{country.ToLower()}-{datasetName}"));

                sw.WriteLine("ProductID\tCoboughtProductID");

                foreach(var item in coboughtProducts)
                {
                    sw.WriteLine("{0}\t{1}", item.ProductID, item.CoboughtProductID);
                }
                sw.Close();
            }
            var model = CreateHomeIndexViewModel();
            return View("Index", model);
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }

        private string GetDataPath(string file)
        {
            return Path.Combine(_webHostEnvironment.ContentRootPath, "wwwroot", "Data", file);
        }
        private HomeIndexViewModel CreateHomeIndexViewModel()
        {
            return new HomeIndexViewModel
            {
                Categories = _db.Categories.Include(category => category.Products),
                GermanyDatasetExists = System.IO.File.Exists(GetDataPath("germany-dataset.txt")),
                UKDatasetExists = System.IO.File.Exists(GetDataPath("uk-dataset.txt")),
                USADatasetExists = System.IO.File.Exists(GetDataPath("usa-dataset.txt")),
                
            };
        }
    }
}
