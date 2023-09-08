from reposClass import GetRepos
# Retrieve the first page of results with 10 results per page



class get_repo_URLS:
  def get_urls(firstPage,github_API_URL):
            
      page =firstPage
      response_code,total_count,repositories = GetRepos.get_repos(page,github_API_URL)


      while (len(repositories) < total_count and response_code==200):
          page += 1
          response_code,ttc,results = GetRepos.get_repos(page,github_API_URL)
          if response_code !=200:
            break
          repositories.extend(results)
        
      csv_file = "repositories"+str(firstPage)+"-"+str(page) +".csv"
      GetRepos.write_repositories_to_csv(repositories, csv_file)

      print(f"Repositories written to {csv_file}")
      return csv_file,page


        

